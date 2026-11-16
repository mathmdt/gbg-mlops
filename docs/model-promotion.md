# Changer de modèle en production et revenir en arrière

L'API sert toujours `gbg-risk@champion`. Un nouvel entraînement **n'y touche pas** :
il enregistre une nouvelle version dans le registre MLflow, avec sa métrique
d'évaluation `gbg_reckless_auc` (ROC-AUC de *Reckless* sur GBG500). Changer de modèle
servi, c'est déplacer l'alias `champion`, et seulement si la nouvelle version est meilleure.

## En 5 commandes

```sh
# 1. Voir la version servie
curl -s localhost:8000/health

# 2. Proposer la version N : promue seulement si son gbg_reckless_auc est meilleur
make promote VERSION=N

# 3. Vérifier que l'API sert la nouvelle version (rechargée automatiquement)
curl -s localhost:8000/health

# 4. En cas de problème en production : revenir à la version précédente
make rollback

# 5. (Démo uniquement) promouvoir sans comparer les métriques
make promote VERSION=N FORCE=1
```

## Que se passe-t-il si le nouveau modèle est moins bon ?

La promotion est **refusée** : `champion` ne bouge pas, l'API continue de servir
l'ancienne version, et le script sort avec le code `2`, ce qui fait échouer un workflow
GitHub Actions. Exemple réel (métriques de démonstration) :

```json
{
  "action": "promote",
  "status": "refused",
  "champion_before": "2",
  "champion_after": "2",
  "candidate_version": "1",
  "metric": "gbg_reckless_auc",
  "candidate_metric": 0.55,
  "champion_metric": 0.62,
  "reason": "candidate 0.5500 <= champion 0.6200: champion kept"
}
```

## Règles

| Situation | Résultat |
|---|---|
| Aucun champion | promue |
| Candidate > champion | promue |
| Candidate ≤ champion | refusée (code 2) |
| Candidate sans `gbg_reckless_auc` | refusée (code 2) |
| Champion sans métrique (modèle factice) | promue |
| Version déjà championne | inchangée (code 0) |
| `--force` | promue sans comparaison |
| Version inexistante, rien à annuler | erreur (code 1) |

## Traçabilité

Chaque promotion écrit sur la version promue les tags `promoted_at` (date UTC) et
`previous_champion` (version remplacée). Un retour arrière remet `champion` sur
`previous_champion` et marque la version retirée avec `rolled_back_at`. Tout est
visible dans l'interface MLflow (http://localhost:5001), onglet *Models*.

## Dans un workflow (réentraînement)

```sh
uv run python scripts/promote_model.py --candidate-version "$VERSION" > decision.json
# code 0 : promue ou inchangée · 2 : refusée · 1 : erreur
jq -r .status decision.json
```

L'API charge le modèle au démarrage : après une promotion, elle doit être redémarrée
(`make reload-api` en dev, déjà inclus dans `make promote` et `make rollback`; un
redéploiement dans Kubernetes).
