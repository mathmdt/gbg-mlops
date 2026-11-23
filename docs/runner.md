# Runner auto-hébergé et déploiement continu (CD)

Le workflow [`.github/workflows/cd.yml`](../.github/workflows/cd.yml) se lance après chaque
CI réussi sur `main`, donc après chaque PR fusionnée dont les tests passent :

1. **publish** (serveurs GitHub) : construit les images `gbg-api` et `gbg-app` pour
   `amd64` et `arm64`, et les publie sur Docker Hub avec deux tags, le SHA du commit et
   `latest`.
2. **deploy** (runner auto-hébergé, sur la machine de démo) :
   - met à jour les Deployments `api` et `app` du namespace `gbg-mlops` avec les images du commit ;
   - attend la fin du déploiement (`rollout status`) ;
   - lance les tests de bout en bout (`tests/e2e`) contre l'API déployée et vérifie que
     la web app répond (`/_stcore/health`) ;
   - en cas d'échec, remet les images qui tournaient avant, seulement sur les
     Deployments que ce déploiement a modifiés.

## Pourquoi un runner auto-hébergé

Les serveurs de GitHub ne peuvent pas joindre le Kubernetes de Docker Desktop, qui
tourne sur un ordinateur portable. Le runner est un petit programme installé sur cet
ordinateur : il demande du travail à GitHub, puis exécute les commandes `kubectl` en
local.

## Prérequis sur la machine de démo

- Docker Desktop avec Kubernetes activé, le contexte `docker-desktop` disponible ;
- la stack déployée une première fois en suivant [`infra/k8s/README.md`](../infra/k8s/README.md)
  (namespace `gbg-mlops`, Secret `gbg-env`, PostgreSQL, MinIO, MLflow, API, web app
  `app.yaml` et un modèle `gbg-risk@champion`) ;
- `kubectl` dans le `PATH` du terminal qui lance le runner.

## Secrets GitHub (une fois)

Dans le repo : **Settings → Secrets and variables → Actions → New repository secret**.

| Nom | Valeur |
|---|---|
| `DOCKERHUB_USERNAME` | le compte Docker Hub qui publie les images |
| `DOCKERHUB_TOKEN` | un *Personal access token* de ce compte (Docker Hub → Account settings → Personal access tokens, droits **Read & Write**) |

Le token n'est jamais écrit dans le code ni dans un fichier : seul GitHub le connaît.

## Installer le runner (une fois, par un administrateur du repo)

Seul un administrateur du repo voit le menu des runners.

1. Sur GitHub : **Settings → Actions → Runners → New self-hosted runner**, choisir
   **macOS** et l'architecture de la machine (**ARM64** pour Apple Silicon, **x64** pour
   Intel).
2. Copier-coller les commandes affichées par GitHub dans un terminal, dans un dossier
   **hors du repo**, par exemple `~/actions-runner`. Elles téléchargent le runner, puis
   `./config.sh --url … --token …` l'enregistre (garder les réponses par défaut).
3. Lancer le runner : `./run.sh`. Il doit afficher `Listening for Jobs`.

Le runner ne travaille que tant que ce terminal est ouvert et l'ordinateur allumé. Pour la
démo, le lancer avant l'oral. Pour qu'il démarre tout seul : `./svc.sh install` puis
`./svc.sh start`.

## Activer le déploiement (une fois le runner installé)

Tant que le runner n'est pas installé, le CD publie seulement les images : le job
`deploy` est désactivé pour ne pas rester bloqué en attente d'une machine. Pour
l'activer : **Settings → Secrets and variables → Actions → onglet Variables → New
repository variable**, nom `DEPLOY_ENABLED`, valeur `true`.

## Vérifier

- **Settings → Actions → Runners** : le runner apparaît avec l'état **Idle**.
- **Actions → CD → Run workflow** relance un déploiement à la main.
- `kubectl --context docker-desktop -n gbg-mlops get deployment api -o wide` affiche
  l'image `…/gbg-api:<sha du dernier commit>`.

## Revenir en arrière à la main

```bash
kubectl --context docker-desktop -n gbg-mlops rollout undo deployment/api
kubectl --context docker-desktop -n gbg-mlops rollout undo deployment/app
```

Pour revenir à un autre **modèle** (et pas à une autre image de l'API) : `make rollback`,
qui déplace l'alias `champion` dans MLflow (voir [`model-promotion.md`](model-promotion.md)).
