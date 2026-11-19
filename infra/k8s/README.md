# Déploiement local Kubernetes — Docker Desktop

## Prérequis

- Docker Desktop avec Kubernetes activé et un nœud Ready
- kubectl et contexte docker-desktop
- Commandes exécutées depuis la racine du dépôt
- Fichier .env configuré selon .env.example
- Environnement Python du projet installé pour les scripts

Ce déploiement crée un registre MLflow distinct de Docker Compose.
Les modèles et données présents dans Compose ne sont pas transférés.

## Images

API : emmaneum/gbg-api:dd41f11, publiée pour amd64 et arm64.
MLflow : image locale, à construire sur chaque machine :

```bash
docker build -f docker/mlflow.Dockerfile -t gbg-mlops/mlflow:dev .
```

Le Deployment MLflow utilise imagePullPolicy: Never.
Les images PostgreSQL et MinIO sont épinglées par digest.

## Installation initiale

Vérifier le cluster :

```bash
kubectl --context docker-desktop get nodes
```

Créer le namespace et le Secret sans écrire les identifiants dans Git :

```bash
kubectl --context docker-desktop apply -f infra/k8s/namespace.yaml

kubectl --context docker-desktop -n gbg-mlops create secret generic gbg-env \
  --from-env-file=.env --dry-run=client -o yaml |
kubectl --context docker-desktop -n gbg-mlops apply -f -
```

Déployer les dépendances dans cet ordre :

```bash
kubectl --context docker-desktop apply -f infra/k8s/postgres.yaml
kubectl --context docker-desktop apply -f infra/k8s/minio.yaml

kubectl --context docker-desktop -n gbg-mlops rollout status deployment/postgres --timeout=180s
kubectl --context docker-desktop -n gbg-mlops rollout status deployment/minio --timeout=180s

kubectl --context docker-desktop apply -f infra/k8s/minio-init.yaml
kubectl --context docker-desktop -n gbg-mlops wait \
  --for=condition=complete job/minio-init --timeout=180s

kubectl --context docker-desktop apply -f infra/k8s/mlflow.yaml
kubectl --context docker-desktop -n gbg-mlops rollout status deployment/mlflow --timeout=300s
```

## Enregistrer un modèle de test

Dans deux terminaux séparés, laisser ces commandes actives :

```bash
kubectl --context docker-desktop -n gbg-mlops port-forward service/mlflow 5001:5000
```

```bash
kubectl --context docker-desktop -n gbg-mlops port-forward service/minio 9002:9000
```

Dans un troisième terminal, depuis la racine du dépôt :

```bash
PYTHONPATH="$PWD/src:$PWD" \
MLFLOW_TRACKING_URI=http://localhost:5001 \
MLFLOW_S3_ENDPOINT_URL=http://localhost:9002 \
.venv/bin/python scripts/register_dummy_model.py
```

Le script enregistre un modèle constant à 0,05.
Il ne remplace pas un vrai modèle champion par défaut.

## Déployer et consulter l'API

```bash
kubectl --context docker-desktop apply -f infra/k8s/api.yaml
kubectl --context docker-desktop -n gbg-mlops rollout status deployment/api --timeout=300s
```

Dans un terminal dédié :

```bash
kubectl --context docker-desktop -n gbg-mlops port-forward service/api 8001:8000
```

Depuis un autre terminal :

```bash
curl --fail --max-time 15 http://localhost:8001/health
```

Documentation interactive : http://localhost:8001/docs
POST /predict exige au moins 300 mesures espacées de 100 ms et le champ units.

Les port-forwards servent aux accès depuis le Mac.
Les services communiquent entre eux directement dans Kubernetes.

## Modèle réel

Enregistrer le CNN et ses artefacts dans ce registre MLflow, puis promouvoir
sa version selon les règles du projet. L'API charge champion au démarrage.
Après un changement d'alias :

```bash
kubectl --context docker-desktop -n gbg-mlops rollout restart deployment/api
kubectl --context docker-desktop -n gbg-mlops rollout status deployment/api --timeout=300s
```

Vérifier ensuite la version effectivement servie avec GET /health.
Relancer le port-forward API si le remplacement du Pod l'a interrompu.

## Ressources et diagnostic

MLflow utilise un worker et une limite mémoire de 2 GiB.
La limite initiale de 1536 MiB a provoqué un OOMKilled sur le Mac de test.
Arrêter les anciens projets inutilisés pour libérer de la mémoire.

```bash
kubectl --context docker-desktop -n gbg-mlops get pods,pvc
kubectl --context docker-desktop -n gbg-mlops logs deployment/mlflow --tail=60
kubectl --context docker-desktop -n gbg-mlops describe pods -l app=mlflow
```

Pour mettre à jour MLflow sur une machine à mémoire limitée, éviter deux
instances simultanées : réduire ses réplicas à zéro, attendre la suppression
du Pod, puis appliquer mlflow.yaml pour relancer une instance.

## Arrêt temporaire

```bash
kubectl --context docker-desktop -n gbg-mlops scale \
  deployment/api deployment/mlflow deployment/minio deployment/postgres --replicas=0
```

Les volumes sont conservés. Arrêter les port-forwards avec Ctrl+C.
Supprimer le namespace ou les PVC peut supprimer les données persistantes.

## Validation effectuée sur Intel

- PostgreSQL, MinIO, MLflow et API : Ready, sans redémarrage au contrôle final
- Volumes PostgreSQL et MinIO : Bound
- Modèle factice version 1 enregistré avec l'alias champion
- GET /health : ready, version 1
- POST /predict : HTTP 200, score proche de 0,05, une fenêtre

Le vrai CNN et l'exécution sur Apple Silicon restent à vérifier.
Prefect et la web app ne sont pas inclus dans ce déploiement.
