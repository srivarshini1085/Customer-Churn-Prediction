# Deploying the dashboard

The Streamlit dashboard (`app.py`) can run as a free public demo on
**Streamlit Community Cloud** (recommended) or **Hugging Face Spaces**.

## How the hosted demo gets a model

Model binaries (`model.joblib`, `survival.joblib`) are deliberately **not** in
git: they are pickles, which can execute code when loaded and only load reliably
with the same library versions. Instead the dashboard **trains on first start**:

* Set `CHURN_AUTO_TRAIN=true`.
* On the first page load, `churn.ui.components.ensure_model()` sees that no model
  can be loaded and runs the normal training pipeline (`python -m churn train`)
  on the bundled dataset `data/Telco-Customer-Churn.csv`.
* Training is cached per server process, so only the first visitor waits. It
  takes about 40 seconds on a 12-core laptop; expect a few minutes on a small
  cloud machine. The model is retrained whenever the hosting platform restarts
  the app (its disk is not persistent), which is fine for a demo.

Because the registry manifests and model cards *are* in git, a fresh clone has a
`latest.json` that points at a version without a `model.joblib`. That is
expected: loading fails, and auto-train creates a new version.

Nothing else needs configuring. The dashboard runs with the local model (no
`CHURN_API_URL`), which is what you want for a single-container demo.

## Before you start (once)

1. Check that the dataset is committed, because training needs it:

   ```bash
   git ls-files data/Telco-Customer-Churn.csv
   ```

   It should print the path. If it prints nothing, add and commit the file.
2. Check that no model binaries are committed:

   ```bash
   git ls-files "models/*.joblib" "models/registry/**/*.joblib"
   ```

   This should print nothing.
3. Push the branch you want to deploy (normally `main`) to GitHub.

## Option A: Streamlit Community Cloud (recommended)

1. Go to <https://share.streamlit.io> and sign in with GitHub. Allow access to the
   repository when asked.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Fill in:
   * **Repository:** `srivarshini1085/Customer-Churn-Prediction`
   * **Branch:** `main`
   * **Main file path:** `app.py`
   * **App URL:** choose a subdomain, for example `customer-churn-demo`.
4. Open **Advanced settings**:
   * **Python version:** `3.12` (the version the Docker image and CI use).
   * **Secrets:** paste exactly this. Root-level secrets are exposed to the app as
     environment variables, which is how `churn.config` reads its settings.

     ```toml
     CHURN_AUTO_TRAIN = "true"
     ```
5. Click **Deploy**. Streamlit installs `requirements.txt` (a few minutes the
   first time), then starts the app.
6. Open the app URL. The first load shows "First start: training the model";
   wait for it to finish. The sidebar then shows the model version.
7. Copy the URL into the README's "Live demo" line.

**If something goes wrong**, open **Manage app** (bottom right of the app) to see
the logs:

* `Dataset not found`: the CSV is not committed (step 1 above).
* The app restarts during training: the free tier ran out of memory or time.
  Reload once; if it keeps happening, use Option B, which has more resources.

## Option B: Hugging Face Spaces (Docker)

1. Create a Space at <https://huggingface.co/new-space>: choose **Docker** as the
   SDK and **Blank** as the template.
2. Clone the Space repository, copy this project into it (without `.venv/`,
   `.git/` and `models/registry/*/*.joblib`), and replace the `Dockerfile` with:

   ```dockerfile
   FROM python:3.12-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   RUN useradd --create-home --uid 1000 user && chown -R user /app
   USER user
   ENV CHURN_AUTO_TRAIN=true
   EXPOSE 7860
   CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
   ```

3. At the top of the Space's `README.md`, add the Space settings block:

   ```yaml
   ---
   title: Customer Churn Prediction
   sdk: docker
   app_port: 7860
   ---
   ```

4. Commit and push to the Space. It builds the image, starts the app and trains
   on first load, the same as Option A.

## Running the same thing locally

```bash
docker compose run --rm api python -m churn train
docker compose up
```

The dashboard is then at <http://localhost:8501> and the API at
<http://localhost:8000/docs>.
