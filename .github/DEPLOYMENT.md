# GitHub Pages Deployment Guide

This repository includes automated deployment of the wave propagation demo to GitHub Pages.

## Quick Setup

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages** (in the left sidebar)
3. Under **Source**, select **GitHub Actions**
4. Save the changes

That's it! The workflow will automatically deploy on the next push.

### 2. Trigger First Deployment

You can trigger the deployment in two ways:

**Option A: Push to main branch**
```bash
git add .
git commit -m "feat: add WebAssembly wave demo"
git push origin main
```

**Option B: Manual workflow dispatch**
1. Go to **Actions** tab
2. Select **Deploy Demo to GitHub Pages** workflow
3. Click **Run workflow** → Select `main` branch → **Run workflow**

### 3. View Your Demo

Once deployed (takes ~2-3 minutes), your demo will be available at:

```
https://<your-username>.github.io/algo-pde/
```

Replace `<your-username>` with your GitHub username.

## Workflow Details

The deployment workflow (`.github/workflows/deploy-demo.yml`) does the following:

1. **Build Stage**:
   - Checks out the repository
   - Sets up Go 1.23
   - Sets up Node.js 20
   - Builds the WASM module from `cmd/acoustics-wasm`
   - Installs npm dependencies
   - Builds the Vite app (`demo/dist/`)
   - Uploads the built artifact

2. **Deploy Stage**:
   - Deploys the artifact to GitHub Pages
   - Updates the deployment URL

### Automatic Triggers

The workflow runs automatically on:
- Push to `main` branch when these paths change:
  - `cmd/acoustics-wasm/**`
  - `demo/**`
  - `poisson/**`
  - `.github/workflows/deploy-demo.yml`

### Manual Triggers

You can also trigger the workflow manually:
- Go to **Actions** tab
- Select the workflow
- Click **Run workflow**

## Monitoring Deployments

1. Go to **Actions** tab to see workflow runs
2. Click on a run to see detailed logs
3. Check **Deployments** section on the right sidebar of your repo

## Troubleshooting

### Workflow Fails on Build

**Check Go version**: Ensure the workflow uses Go 1.23+
```yaml
go-version: '1.23'
```

**Check WASM build**: Verify locally with:
```bash
just wasm
```

### Workflow Fails on Deploy

**Check Pages Settings**:
- Ensure GitHub Pages source is set to "GitHub Actions"
- Check if Pages is enabled for your repository

**Check Permissions**:
The workflow needs these permissions (already configured):
```yaml
permissions:
  contents: read
  pages: write
  id-token: write
```

### Demo Loads but Shows Errors

**Check Base Path**: Vite config uses relative base (`./`) for GitHub Pages:
```typescript
base: './'
```

**Check Asset Loading**: Browser console will show if WASM/JS files fail to load

### Force Redeploy

If the deployment seems stuck:
1. Go to **Actions** → **Deploy Demo to GitHub Pages**
2. Click **Run workflow** → **Run workflow**
3. Wait 2-3 minutes for completion

## Local Testing of Production Build

Before pushing, test the production build locally:

```bash
# Build WASM and demo
just demo-build

# Serve the built files
cd demo/dist
python3 -m http.server 8000

# Open http://localhost:8000
```

This ensures the production build works before deployment.

## Updating the Demo

After making changes to the demo:

```bash
# Test locally first
just demo-dev

# Build and test production
just demo-build
cd demo/dist && python3 -m http.server 8000

# Commit and push
git add .
git commit -m "update: improve wave demo performance"
git push origin main

# Workflow automatically deploys
```

## Cost & Limits

GitHub Pages is free for public repositories with these limits:
- **Bandwidth**: 100 GB/month
- **Build time**: 10 minutes per workflow
- **Storage**: 1 GB

The demo is ~4 MB, so this should be more than sufficient.

## Custom Domain (Optional)

To use a custom domain:

1. Add a `CNAME` file to `demo/public/`:
   ```
   demo.yourdomain.com
   ```

2. Configure DNS:
   - Add CNAME record: `demo.yourdomain.com` → `<username>.github.io`

3. Update GitHub Pages settings:
   - Settings → Pages → Custom domain → Enter `demo.yourdomain.com`

## Security Notes

The workflow:
- Uses pinned action versions (`@v4`, `@v5`)
- Minimal permissions (read contents, write pages)
- No secrets required
- Sandboxed environment

## Support

For issues with:
- **Workflow**: Check `.github/workflows/deploy-demo.yml`
- **Build**: Check `demo/vite.config.ts` and `justfile`
- **Demo**: Check `demo/README.md`

For GitHub Pages issues, see: https://docs.github.com/pages
