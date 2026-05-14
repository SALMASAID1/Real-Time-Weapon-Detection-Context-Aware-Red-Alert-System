# Jalon 4 : Système d'Alertes Multimodales et Tableau de Bord en Temps Réel

Ce document détaille l'implémentation du Jalon 4, qui se concentre sur la construction de l'infrastructure de réponse en temps réel : alertes multi-canaux, moteur d'orchestration d'inférence, backend FastAPI et tableau de bord de surveillance React.

---

## Étape 1 : Dispatch d'Alertes — Moteur de Notification Multi-Canaux
**Fichier :** `src/threat_logic/alert_dispatcher.py`

Lorsque le Scoreur de Menaces élève une détection au niveau "HIGH", l'AlertDispatcher déclenche des notifications à travers trois canaux indépendants simultanément.

**Actions Clés :**
1. **Bot Telegram (Asynchrone) :** Utilise l'API async `python-telegram-bot` v20. Envoie un message Markdown formaté avec l'ID de la caméra, l'horodatage, le nom de la classe, la confiance, le score composite et l'IoU de proximité. Si un JPEG Grad-CAM est disponible, il est envoyé en pièce jointe photo. Dispatché en tâche « fire-and-forget » via `asyncio.create_task()` pour ne jamais bloquer la boucle d'inférence en attendant la réponse de l'API Telegram.
2. **Alerte Audio (Pygame) :** Joue un fichier WAV pré-chargé (`data/audio/alert.wav`) via `pygame.mixer.Sound`. Pygame est initialisé une seule fois au démarrage — pas à chaque alerte — pour éviter les pics de latence.
3. **Journal d'Événements (JSONL) :** Ajoute un enregistrement JSON structuré dans `data/logs/threats.jsonl`. Champs : `event_id`, `timestamp`, `camera_id`, `class_name`, `confidence`, `composite_score`, `bbox`, `dispatched`, `acknowledged`.
4. **Déduplication par Refroidissement :** Chaque détection se voit attribuer une clé spatiale basée sur sa position dans la grille. Si la même clé se déclenche dans les 30 secondes, Telegram et l'audio sont ignorés (prévention du spam de notifications), mais l'événement est toujours enregistré.

---

## Étape 2 : Moteur d'Inférence — Orchestration des Trames
**Fichier :** `src/inference/engine.py`

L'InferenceEngine se situe entre la source caméra et le diffuseur WebSocket. Il possède la boucle de trames et gère tous les composants du pipeline.

**Actions Clés :**
1. **Stratégie à Double Cadence :** L'inférence SAHI par tuiles complète s'exécute toutes les N trames (défaut N=3). L'inférence légère en passage unique gère toutes les autres trames. Cela atteint ~30 FPS de débit effectif contre ~6 FPS avec SAHI sur chaque trame.
2. **Pipeline à Double Flux :** Le flux Armes (YOLO-Swin Hybride) et le flux Mains (YOLOv8) s'exécutent en parallèle sur chaque trame. Leurs détections sont fusionnées avant le scoring de menace.
3. **Intégration du Score de Menace :** Toutes les détections sont transmises au `ThreatScorer`, qui calcule les scores composites et identifie la détection de plus haute menace.
4. **Déclenchement d'Alertes :** En cas de menace HIGH, le moteur dispatche les alertes via `AlertDispatcher` et génère optionnellement une superposition de carte de chaleur Grad-CAM.
5. **Diffusion par File d'Attente :** Les objets `FrameResult` complétés (JPEG encodé en base64 + détections + niveau de menace + Grad-CAM) sont poussés dans une `asyncio.Queue`. Une tâche en arrière-plan dans `main.py` consomme cette file et diffuse à tous les clients WebSocket connectés.

---

## Étape 3 : Backend FastAPI — Surface API
**Fichier :** `src/api/main.py` et routeurs dans `src/api/routers/`

Le backend fournit à la fois du streaming WebSocket et des points d'accès REST.

**Actions Clés :**
1. **Gestion du Cycle de Vie :** Tous les modèles, pipelines et le moteur d'inférence sont initialisés au démarrage de FastAPI via `@asynccontextmanager`. À l'arrêt, le moteur est stoppé et les tâches en arrière-plan sont annulées proprement.
2. **Streaming WebSocket** (`/ws/stream/{camera_id}`) : Un `ConnectionManager` maintient un ensemble de connexions actives par caméra. La tâche de diffusion envoie le JSON `FrameResult` sérialisé à tous les clients connectés.
3. **API REST des Menaces** (`/api/threats`) : CRUD complet sur le journal d'événements JSONL — listing paginé avec filtres (caméra, classe, niveau, horodatage, acquitté), récupération d'événement unique, acquittement et suppression.
4. **API REST des Paramètres** (`/api/settings`) : GET/PUT pour la configuration en temps réel. PUT propage les changements (seuils, cadence SAHI) directement au dictionnaire de paramètres de l'InferenceEngine en cours d'exécution.
5. **Schémas Pydantic** (`src/api/schemas/detection.py`) : Contrats de données stricts — `Detection`, `ThreatEvent`, `WebSocketFrame`, `SystemSettings` — garantissant la sécurité des types entre le backend Python et le frontend React.

---

## Étape 4 : Tableau de Bord React — Interface de Surveillance
**Répertoire :** `ui/src/`

Une application React (Vite) à page unique fournissant trois vues pour les opérateurs de sécurité.

**Actions Clés :**
1. **LiveMonitor** (`pages/LiveMonitor.jsx`) : La page principale de surveillance. Possède la connexion WebSocket via le hook `useWebSocket`. Affiche le flux vidéo en direct sur un `VideoCanvas`, montre les détections en temps réel dans un panneau latéral, et présente une bannière `RedAlertBanner` en superposition lors de menaces HIGH.
2. **ThreatHistory** (`pages/ThreatHistory.jsx`) : Journal historique des événements récupéré via `GET /api/threats`. Supporte la pagination, le filtrage par niveau de menace et classe d'arme, et un bouton de rafraîchissement.
3. **Settings** (`pages/Settings.jsx`) : Configuration en temps réel via `ThresholdPanel` — curseurs pour le seuil de confiance, le seuil IoU, la cadence SAHI, etc. Soumission via `PUT /api/settings`.
4. **Hook WebSocket** (`hooks/useWebSocket.js`) : Gère le cycle de vie complet du WebSocket avec reconnexion à backoff exponentiel (1s → 2s → 4s → ... → 30s plafond). Parse les trames JSON entrantes et expose les états `lastMessage` et `isConnected`.
5. **Système de Design** (`index.css`) : Système de design sombre de 15 Ko avec variables CSS, cartes glassmorphism et mises en page responsives.

---

## Scripts du Pipeline
*   `src/api/main.py` : Point d'entrée de l'application et gestion du cycle de vie.
*   `src/api/routers/stream.py` : Point d'accès WebSocket et ConnectionManager.
*   `src/api/routers/threats.py` : API REST CRUD des événements de menace.
*   `src/api/routers/settings.py` : GET/PUT des paramètres en temps réel avec propagation au moteur.
*   `src/threat_logic/alert_dispatcher.py` : Dispatch de notifications multi-canaux.
*   `src/inference/engine.py` : Boucle de trames, inférence à double cadence, diffusion par file d'attente.
