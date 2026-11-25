# Exploitation : lancer, entraîner, déployer, revenir en arrière

Ce document renvoie aux procédures détaillées plutôt que de les recopier.

## Lancer la pile de développement

```sh
make install        # une fois : .venv avec les dépendances
make up             # Postgres, MinIO, MLflow, API (crée .env si absent)
make dummy-model    # enregistre un modèle factice en champion et recharge l'API
curl localhost:8000/health
make down           # arrêt ; les volumes sont conservés
```

Pour mettre un vrai modèle en `champion` sans réentraîner, importer le modèle exporté
(dossier `exported_model/` avec `weights.pt`, `config.json`, `normalization.json`) :

```sh
uv sync --group training
uv run python scripts/register_exported_model.py --model-dir exported_model
make reload-api
```

## Préparer les données et entraîner

```sh
make data           # data/raw/ → data/processed/
make train          # entraîne le CNN, l'enregistre dans MLflow (champion inchangé)
make compare-runs   # compare 3 variantes dans MLflow
make dag            # tout enchaîner avec Prefect : ingest → prepare → train → evaluate → promote
```

L'entraînement demande `uv sync --group training` ; PyTorch n'a pas de wheel pour Mac Intel
(entraîner sur Apple Silicon ou dans un conteneur Linux).

## Changer le modèle servi et revenir en arrière

```sh
curl -s localhost:8000/health   # version servie
make promote VERSION=N          # promue seulement si gbg_reckless_auc est meilleur
make rollback                   # revient à la version précédente
```

Une promotion refusée laisse `champion` en place et sort avec le code 2. Règles complètes
et traçabilité : [`model-promotion.md`](model-promotion.md).

## Déployer sur Kubernetes (Docker Desktop)

La procédure complète, dans l'ordre (namespace, secret, image MLflow locale, PostgreSQL,
MinIO, MLflow, API, web app, modèle réel, ports) est dans
[`../infra/k8s/README.md`](../infra/k8s/README.md). Accès depuis le Mac :

| Service | Commande | URL |
|---|---|---|
| Web app | `kubectl --context docker-desktop -n gbg-mlops port-forward service/app 8502:8501` | http://localhost:8502 |
| API | `kubectl --context docker-desktop -n gbg-mlops port-forward service/api 8001:8000` | http://localhost:8001/docs |
| MLflow | `kubectl --context docker-desktop -n gbg-mlops port-forward service/mlflow 5001:5000` | http://localhost:5001 |

Après un changement d'alias `champion`, redémarrer l'API pour qu'elle charge la nouvelle
version :

```sh
kubectl --context docker-desktop -n gbg-mlops rollout restart deployment/api
kubectl --context docker-desktop -n gbg-mlops rollout status deployment/api --timeout=300s
```

## Déploiement continu

Le CD publie les images après chaque CI verte sur `main`, puis déploie par un runner
auto-hébergé si `DEPLOY_ENABLED=true`. Installation du runner, secrets et variable :
[`runner.md`](runner.md). Relancer à la main : **Actions → CD → Run workflow** sur `main`.

## Diagnostic

| Symptôme | À vérifier |
|---|---|
| `/health` répond 503 | MLflow est-il prêt, `champion` existe-t-il ? Puis redémarrer l'API. C'est aussi ce qui arrive après un redémarrage du Mac. |
| Pod MLflow `OOMKilled` | Mémoire de Docker Desktop (le manifeste limite MLflow à 2 Gio) ; fermer les projets Docker inutiles. |
| MLflow `ErrImageNeverPull` | L'image `gbg-mlops/mlflow:dev` doit exister en local : `docker build -f docker/mlflow.Dockerfile -t gbg-mlops/mlflow:dev .` |
| Port-forward coupé | Le relancer après chaque remplacement de pod. |
| Promotion refusée | Normal si le candidat n'est pas meilleur : lire `reason` dans la sortie, ou `FORCE=1` pour une démo. |
| Scores différents de la référence | Vérifier `units=deg_s` pour GBG500 et les fichiers utilisés. |

Commandes utiles :

```sh
kubectl --context docker-desktop -n gbg-mlops get pods,pvc
kubectl --context docker-desktop -n gbg-mlops logs deployment/api --tail=80
kubectl --context docker-desktop -n gbg-mlops describe pods -l app=api
```

Ne pas supprimer le namespace ni les volumes pour débloquer une situation : le modèle et
les données persistantes seraient perdus.
