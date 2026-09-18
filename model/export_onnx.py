"""
export_onnx.py — Constrói bemtevi_detector.onnx a partir de params.npz.

Roda em processo SEPARADO do treino (train_model.py o invoca) porque
sklearn e onnx entram em conflito de DLL no Windows se carregados juntos.
Aqui NÃO importamos sklearn — apenas numpy e onnx.

Grafo:  features(1,28) -> Sub(mean) -> Div(scale) -> Gemm(W,b) -> Sigmoid -> probability(1,1)
A padronização e a regressão logística ficam embutidas no próprio modelo.
"""
import os
import numpy as np
from onnx import helper, TensorProto, numpy_helper, checker
import onnx

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "params.npz")
ONNX_PATH = os.path.join(HERE, "bemtevi_detector.onnx")


def build_onnx(mean, scale, W0, b0, W1, b1, W2, b2, threshold):
    """MLP: features -> Sub -> Div -> [Gemm+Relu]x2 -> Gemm -> Sigmoid."""
    n = F.N_FEATURES
    inits = [
        numpy_helper.from_array(mean.astype(np.float32), "feat_mean"),
        numpy_helper.from_array(scale.astype(np.float32), "feat_scale"),
        numpy_helper.from_array(W0.astype(np.float32), "W0"),
        numpy_helper.from_array(b0.astype(np.float32), "B0"),
        numpy_helper.from_array(W1.astype(np.float32), "W1"),
        numpy_helper.from_array(b1.astype(np.float32), "B1"),
        numpy_helper.from_array(W2.astype(np.float32), "W2"),
        numpy_helper.from_array(b2.astype(np.float32), "B2"),
    ]
    X = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, n])
    P = helper.make_tensor_value_info("probability", TensorProto.FLOAT, [1, 1])
    nodes = [
        helper.make_node("Sub", ["features", "feat_mean"], ["centered"]),
        helper.make_node("Div", ["centered", "feat_scale"], ["z"]),
        helper.make_node("Gemm", ["z", "W0", "B0"], ["a0"]),
        helper.make_node("Relu", ["a0"], ["h0"]),
        helper.make_node("Gemm", ["h0", "W1", "B1"], ["a1"]),
        helper.make_node("Relu", ["a1"], ["h1"]),
        helper.make_node("Gemm", ["h1", "W2", "B2"], ["logit"]),
        helper.make_node("Sigmoid", ["logit"], ["probability"]),
    ]
    graph = helper.make_graph(
        nodes, "bemtevi_detector", [X], [P], initializer=inits,
        doc_string=(f"Detector de bem-te-vi (MLP). threshold={threshold:.4f}. "
                    f"anomalia se probability > threshold"))
    model = helper.make_model(graph, producer_name="ponderada-bemtevi",
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    checker.check_model(model)
    return model


def _mlp_forward(x, mean, scale, W0, b0, W1, b1, W2, b2):
    z = (x - mean) / scale
    h0 = np.maximum(z @ W0 + b0, 0.0)
    h1 = np.maximum(h0 @ W1 + b1, 0.0)
    return 1.0 / (1.0 + np.exp(-(h1 @ W2 + b2)))


def main():
    if not os.path.exists(PARAMS_PATH):
        raise SystemExit("params.npz não encontrado. Rode train_model.py primeiro.")
    d = np.load(PARAMS_PATH)
    mean, scale, threshold = d["mean"], d["scale"], float(d["threshold"])
    W0, b0, W1, b1, W2, b2 = d["W0"], d["b0"], d["W1"], d["b1"], d["W2"], d["b2"]

    model = build_onnx(mean, scale, W0, b0, W1, b1, W2, b2, threshold)
    onnx.save(model, ONNX_PATH)
    print(f"  ONNX salvo -> {ONNX_PATH}")

    # verificação numérica opcional (onnxruntime pode faltar DLL — é tolerado)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        x = np.random.RandomState(0).randn(1, F.N_FEATURES).astype(np.float32)
        expected = float(_mlp_forward(x, mean, scale, W0, b0, W1, b1, W2, b2).ravel()[0])
        got = float(sess.run(None, {"features": x})[0].ravel()[0])
        print(f"  verificacao ONNX: got={got:.6f} esperado={expected:.6f}")
    except Exception as e:
        print(f"  (verificacao onnxruntime pulada: {type(e).__name__})")


if __name__ == "__main__":
    main()
