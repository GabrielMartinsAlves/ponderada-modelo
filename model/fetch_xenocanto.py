"""
fetch_xenocanto.py — Baixa gravações reais do xeno-canto (API v3).

Positivos: bem-te-vi  (Pitangus sulphuratus), Brasil.
Negativos: amostra DIVERSA de outras aves brasileiras (exclui o gênero
           Pitangus), com limite por espécie para garantir variedade.

Não baixamos "todas as aves do Brasil" (seriam centenas de milhares de
arquivos / dezenas de GB): para um classificador binário basta um conjunto
balanceado de positivos + uma amostra representativa de negativos.

Uso:
    set XC_KEY=sua_chave        (Windows)  |  export XC_KEY=...  (Linux/Mac)
    python fetch_xenocanto.py --pos 120 --neg 350

A chave NUNCA é impressa nem gravada em disco. Os áudios vão para
../data/pos e ../data/neg, com um manifest.csv em cada pasta.
"""
import os
import sys
import csv
import time
import random
import argparse
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
API = "https://xeno-canto.org/api/3/recordings"

POS_QUERY = 'sp:"pitangus sulphuratus" cnt:brazil grp:birds'
# negativos: aves do Brasil em geral (filtramos Pitangus depois)
NEG_QUERY = 'cnt:brazil grp:birds'


def api_get(query, key, page=1, per_page=100):
    r = requests.get(API, params={"query": query, "key": key,
                                  "per_page": per_page, "page": page},
                     timeout=60)
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message", r.text[:200])
        except Exception:
            msg = r.text[:200]
        raise SystemExit(f"Erro da API (HTTP {r.status_code}): {msg}")
    return r.json()


def download_file(url, dest, session):
    try:
        with session.get(url, stream=True, timeout=120,
                          allow_redirects=True) as resp:
            if resp.status_code != 200:
                return False
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
        return os.path.getsize(dest) > 1024  # descarta downloads vazios
    except Exception:
        return False


def collect_positives(key, n_target):
    """Todas as gravações de bem-te-vi até n_target (percorre páginas)."""
    recs, page = [], 1
    while len(recs) < n_target:
        data = api_get(POS_QUERY, key, page=page)
        recs.extend(data["recordings"])
        num_pages = int(data["numPages"])
        print(f"  [pos] pagina {page}/{num_pages} -> {len(recs)} gravacoes")
        if page >= num_pages:
            break
        page += 1
        time.sleep(0.3)
    return recs[:n_target]


def collect_negatives(key, n_target, max_per_species=3):
    """Amostra diversa de aves brasileiras que NÃO são Pitangus."""
    data = api_get(NEG_QUERY, key, page=1)
    num_pages = int(data["numPages"])
    print(f"  [neg] {data['numRecordings']} gravacoes em {num_pages} paginas; amostrando...")
    pages = list(range(1, num_pages + 1))
    random.shuffle(pages)

    chosen, per_species = [], {}
    for pg in pages:
        if len(chosen) >= n_target:
            break
        d = data if pg == 1 else api_get(NEG_QUERY, key, page=pg)
        recs = d["recordings"]
        random.shuffle(recs)
        for rec in recs:
            if len(chosen) >= n_target:
                break
            if rec.get("gen", "").lower() == "pitangus":       # exclui bem-te-vi
                continue
            sp = f"{rec.get('gen','')} {rec.get('sp','')}".lower()
            if per_species.get(sp, 0) >= max_per_species:        # diversidade
                continue
            if not rec.get("file"):                              # espécie restrita
                continue
            per_species[sp] = per_species.get(sp, 0) + 1
            chosen.append(rec)
        time.sleep(0.3)
    print(f"  [neg] espécies distintas amostradas: {len(per_species)}")
    return chosen


def save_set(recs, folder):
    os.makedirs(folder, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "ponderada-bemtevi/1.0"})
    manifest = os.path.join(folder, "manifest.csv")
    ok = 0
    with open(manifest, "w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        w.writerow(["id", "gen", "sp", "en", "q", "length", "file"])
        for rec in recs:
            xc_id = rec["id"]
            dest = os.path.join(folder, f"XC{xc_id}.mp3")
            if os.path.exists(dest) and os.path.getsize(dest) > 1024:
                ok += 1
                w.writerow([xc_id, rec.get("gen"), rec.get("sp"),
                            rec.get("en"), rec.get("q"), rec.get("length"),
                            rec.get("file")])
                continue
            if download_file(rec["file"], dest, session):
                ok += 1
                w.writerow([xc_id, rec.get("gen"), rec.get("sp"),
                            rec.get("en"), rec.get("q"), rec.get("length"),
                            rec.get("file")])
                print(f"    ok XC{xc_id} ({rec.get('gen')} {rec.get('sp')})")
            else:
                print(f"    falhou XC{xc_id}")
            time.sleep(0.25)
    print(f"  -> {ok} arquivos em {folder}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("XC_KEY", ""))
    ap.add_argument("--pos", type=int, default=120, help="nº de gravações de bem-te-vi")
    ap.add_argument("--neg", type=int, default=350, help="nº de gravações negativas")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.key:
        raise SystemExit("Faltou a API key. Use XC_KEY=... ou --key ... "
                         "(pegue em https://xeno-canto.org na sua conta).")
    random.seed(args.seed)

    print("== Coletando POSITIVOS (bem-te-vi) ==")
    pos = collect_positives(args.key, args.pos)
    print(f"  encontrados: {len(pos)}")

    print("== Coletando NEGATIVOS (outras aves BR) ==")
    neg = collect_negatives(args.key, args.neg)
    print(f"  selecionados: {len(neg)}")

    print("== Baixando áudios ==")
    save_set(pos, os.path.join(DATA, "pos"))
    save_set(neg, os.path.join(DATA, "neg"))
    print("== Concluído. Agora rode: python build_real_dataset.py ==")


if __name__ == "__main__":
    main()
