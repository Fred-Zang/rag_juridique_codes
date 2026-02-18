# 0. (venv_linux) activié et dansd rag_bench$
source .venv_linux/bin/activate

# 1. lancer run.py par CLI

PYTHONPATH=src python -m rag_bench.run --config configs/benchmark_cdtravail.yaml


# lancer ES par CLI

1. lancer le compose dnas 1 terminal

docker compose -f docker-compose.elasticsearch.yml up

2. lancer dans un 2eme terminal
- verif EC localhost actif (car ça prend du temps)

curl -s "http://localhost:9200/_cluster/health?pretty"

3. lancer run.py dans un autre terminal
source .venv_linux/bin/activate
PYTHONPATH=src python -m rag_bench.run --config configs/benchmark_cdtravail.yaml

4. vérification Post-run
# Santé cluster
curl -s "http://localhost:9200/_cluster/health?pretty"

# Liste des index (taille, docs, status)
curl -s "http://localhost:9200/_cat/indices?v"

# Nombre de documents dans notre index
curl -s "http://localhost:9200/rag_bench_cdtravail/_count?pretty"

# Aperçu de 3 docs (pour vérifier le mapping/champs)
curl -s "http://localhost:9200/rag_bench_cdtravail/_search?size=3&pretty"



3. bis (bis = avant modif et plus actuel à présent)
- lancer le run Elastic_SEARCH

PYTHONPATH=src python -m rag_bench.benchmark_elasticsearch \
  --config configs/benchmark_cdtravail.yaml \
  --es-host http://localhost:9200 \
  --es-index rag_bench_cdtravail
  

