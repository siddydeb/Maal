# Wan Studio — Architecture (Mermaid)

Draw.io–style views of the **current** Wan Studio system: deployment, components, data, and sequential flows.

> **Media policy (current):** originals and thumbs live in **MongoDB Atlas GridFS only**. No local lasting gens on the GPU PC. Openinary / R2 was explored and **cancelled** — slow library load via GridFS is accepted.

**How to view:** open this file in Cursor / GitHub / any Mermaid preview.

Repo: [wellswenger-svg/Maal](https://github.com/wellswenger-svg/Maal) · URLs: [`PRODUCTION_URLS.md`](PRODUCTION_URLS.md)

---

## Table of diagrams

1. [System context](#1-system-context)
2. [Deployment topology](#2-deployment-topology)
3. [GPU PC edge](#3-gpu-pc-edge)
4. [Component architecture](#4-component-architecture)
5. [Data model](#5-data-model)
6. [Auth unlock](#6-auth-unlock)
7. [Job generate — full sequence](#7-job-generate--full-sequence)
8. [AI engine flow](#8-ai-engine-flow)
9. [ComfyUI contract](#9-comfyui-contract)
10. [Library + thumbs (GridFS)](#10-library--thumbs-gridfs)
11. [Delete](#11-delete)
12. [Admin ops](#12-admin-ops)
13. [Scrub / zero-residue](#13-scrub--zero-residue)

---

## 1. System context

```mermaid
flowchart TB
    subgraph Users
        U[Owner / Admin browser]
        T[Tester browser]
    end

    subgraph Cloud["Public cloud"]
        SPA[Vercel SPA]
        API[Render API]
        MONGO[(MongoDB Atlas<br/>wan_studio + GridFS media)]
    end

    subgraph GPUPC["GPU PC — always on"]
        CF[cloudflared tunnels]
        COMFY[ComfyUI :8188]
        GA[gpu_agent :8799]
        WD[wan_stack_watchdog]
    end

    U --> SPA
    T --> SPA
    SPA -->|VITE_API_URL REST + media| API
    API --> MONGO
    API -->|COMFYUI_URL| CF
    CF --> COMFY
    CF --> GA
    WD -.-> CF
    WD -.-> COMFY
    WD -.-> GA
```

---

## 2. Deployment topology

```mermaid
flowchart LR
    subgraph Vercel
        FE[React SPA]
    end

    subgraph Render
        UV[uvicorn backend.main:app]
    end

    subgraph Atlas
        DB[(wan_studio)]
        GFS[(GridFS bucket media)]
    end

    subgraph Home["GPU PC"]
        C8188[Comfy :8188]
        G8799[gpu_agent :8799]
        TUN[cloudflared]
    end

    FE --> UV
    UV --> DB
    UV --> GFS
    UV --> TUN
    TUN --> C8188
    TUN --> G8799
```

| Service | URL |
|---------|-----|
| Frontend | https://frontend-six-chi-37.vercel.app |
| API | https://wan-studio-api.onrender.com |
| GitHub | https://github.com/wellswenger-svg/Maal |

---

## 3. GPU PC edge

```mermaid
flowchart TB
    W[wan_stack_watchdog + autostart]
    P8188[":8188 ComfyUI"]
    P8799[":8799 gpu_agent"]
    TUN[cloudflared named/quick tunnel]
    R1[COMFYUI_URL on Render]
    R2[GPU_AGENT_URL for Controls]

    W --> P8188
    W --> P8799
    W --> TUN
    TUN --> R1
    TUN --> R2
    P8188 --- TUN
    P8799 --- TUN
```

No lasting gens on this PC — Comfy artifacts are scrubbed after each job.

---

## 4. Component architecture

```mermaid
flowchart TB
    subgraph Frontend["frontend/"]
        APP[App.jsx]
        APIJS[api.js]
        APP --> APIJS
    end

    subgraph Backend["backend/"]
        MAIN[main.py]
        OWN[owners.py]
        DB[(db.py GridFS)]
        PN[prompt_fix]
        AE[ai_engine/]
        CC[comfy_client]
        SCR[scrub]
        OPS[ops_remote]
        MAIN --> OWN
        MAIN --> PN
        MAIN --> AE
        MAIN --> CC
        MAIN --> DB
        MAIN --> OPS
        AE --> CC
        CC --> SCR
    end

    APIJS --> MAIN
    DB --> MONGO[(Atlas)]
    CC --> COMFY[ComfyUI]
```

```mermaid
flowchart LR
    REQ[GenerateRequest] --> PL[planner]
    PL --> PLAN[ExecutionPlan]
    PLAN --> REG[registry]
    REG --> MM[models]
    MM --> ENG[runtime]
    ENG --> POST[post]
    POST --> OUT[bytes → GridFS]
```

---

## 5. Data model

```mermaid
erDiagram
    OWNER ||--o{ GENERATION : owns
    OWNER ||--o{ JOB : owns
    GENERATION ||--|| GRIDFS_FILE : gridfs_id
    GENERATION ||--o| THUMB_GFS : thumb_gridfs_id
    JOB ||--o| GRIDFS_INPUT : input_gridfs_id

    GENERATION {
        ObjectId id
        string kind
        string prompt
        string owner
        ObjectId gridfs_id
        object meta
    }
    GRIDFS_FILE {
        ObjectId id
        binData bytes
    }
```

---

## 6. Auth unlock

```mermaid
sequenceDiagram
    actor U as User
    participant SPA as SPA
    participant API as API
    participant OWN as owners.py

    U->>SPA: PIN
    SPA->>API: POST /api/auth/unlock
    API->>OWN: bcrypt check WAN_PINS
    OWN-->>API: owner id
    API-->>SPA: session
```

---

## 7. Job generate — full sequence

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant SPA as SPA
    participant API as FastAPI
    participant DB as Mongo+GridFS
    participant PN as prompt_fix
    participant AE as ai_engine
    participant CC as ComfyClient
    participant COMFY as ComfyUI
    participant SCR as scrub

    U->>SPA: mode + prompt + image
    SPA->>API: POST /api/jobs
    API->>PN: normalize_prompt
    API->>DB: job queued + input GridFS
    API-->>SPA: job id
    SPA->>API: poll GET /api/jobs/id

    API->>AE: run
    AE->>CC: generate
    CC->>COMFY: upload + prompt + WS wait
    COMFY-->>CC: output bytes
    CC->>SCR: wipe wan_* on disk
    CC-->>API: bytes
    API->>DB: GridFS + generations doc
    API->>DB: job done
    SPA->>API: GET /api/media/id/thumb
    API->>DB: thumb GridFS or build
    API-->>SPA: JPEG
```

---

## 8. AI engine flow

```mermaid
flowchart TD
    A[main._execute_generation] --> B[ai_engine.run]
    B --> C[Planner ± VLM]
    C --> D[Registry + models]
    D --> E[Comfy execute]
    E --> F[Post]
    F --> G[GridFS store]
    E -.-> R[Recovery retry]
    R -.-> D
```

---

## 9. ComfyUI contract

```mermaid
sequenceDiagram
    participant CC as ComfyClient
    participant C as ComfyUI
    CC->>C: POST /upload/image
    CC->>C: POST /prompt
    CC->>C: WS wait
    CC->>C: GET /history + /view
    CC->>C: scrub wan_*
```

---

## 10. Library + thumbs (GridFS)

```mermaid
sequenceDiagram
    actor U as User
    participant SPA as SPA
    participant API as API
    participant GFS as GridFS

    U->>SPA: Open library
    SPA->>API: GET /api/generations
    API-->>SPA: list + /api/media/.../thumb
    SPA->>API: GET /api/media/id/thumb
    alt cached thumb
        API->>GFS: thumb bytes
    else cold
        API->>GFS: full original
        API->>API: Pillow/ffmpeg
        API->>GFS: cache thumb
    end
    API-->>SPA: JPEG
```

---

## 11. Delete

```mermaid
sequenceDiagram
    SPA->>API: DELETE /api/generations/id
    API->>API: delete GridFS original + thumb
    API->>API: delete Mongo doc
```

---

## 12. Admin ops

```mermaid
sequenceDiagram
    actor A as Admin
    participant SPA as Controls
    participant API as API
    participant R as Render API
    participant GA as gpu_agent

    A->>SPA: Restart API → API → R
    A->>SPA: Set tunnel → API → R COMFYUI_URL
    A->>SPA: Restart GPU → API → GA → Comfy
```

---

## 13. Scrub / zero-residue

```mermaid
flowchart TB
    subgraph Lasting["Lasting (cloud)"]
        M[Mongo metadata]
        G[GridFS media]
    end
    subgraph Ephemeral["Must not linger on GPU PC"]
        I[Comfy input/output/temp wan_*]
    end
    JOB --> I
    JOB --> FETCH[API fetch bytes]
    FETCH --> G
    FETCH --> SCRUB[scrub local]
    SCRUB --> I
```

---

## Related

| Doc | Role |
|-----|------|
| [`TECHNICAL.md`](TECHNICAL.md) | Module detail |
| [`AI_ENGINE.md`](AI_ENGINE.md) | Planner / VRAM |
| [`PRODUCTION_URLS.md`](PRODUCTION_URLS.md) | Deploy URLs |
| [`media.md`](media.md) | Where files live |
