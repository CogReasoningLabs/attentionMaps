# Google Drive workspace uploads

The WORKSPACE tab can upload a completed Step 3 run directory, including its
clean JSONL, deduplication audit, and manifest. File transfers are resumable,
recursive directory structure is preserved, and Shared Drive requests include
`supportsAllDrives=true`.

## One-time setup

1. Enable the Google Drive API in the Google Cloud project.
2. Choose one authentication method:
   - Configure Application Default Credentials (ADC). For local user credentials,
     run `gcloud auth application-default login --scopes=https://www.googleapis.com/auth/drive.file,https://www.googleapis.com/auth/cloud-platform`.
   - For an existing service-account credential, set
     `GOOGLE_APPLICATION_CREDENTIALS` to its JSON path. Do not put credential JSON
     in this repository.
   - For interactive teammate login, create a **Desktop app** OAuth client,
     download its JSON outside the repository, and select **Browser OAuth** in
     Streamlit. The first upload opens Google sign-in and creates a local token
     cache with user-only file permissions.
3. Share the destination folder or Shared Drive with the authenticated user or
   service-account email.
4. Optionally set the common team destination:

   ```bash
   export ATTENTION_MAPS_DRIVE_PARENT_ID="your-folder-id"
   ```

An API key alone cannot authorize a private Drive write. The uploader defaults
to the narrow `drive.file` OAuth scope. Enable **Use full Drive scope** only when
the team destination cannot be accessed with the narrower scope.

## Streamlit

Run the cleaning gates in **WORKSPACE**. After Step 3, expand **Upload completed
workspace to Google Drive**, enter the destination folder ID, and start the
upload. Both the browser and the Streamlit terminal show progress.

Browser OAuth is suitable only when Streamlit runs on the same local computer
as the browser. Use ADC or a service account on a shared/headless server.

## Command line

The same uploader is available independently of Streamlit:

```bash
python scripts/upload_to_google_drive.py path/to/file-or-folder \
  --parent-folder-id "your-folder-id"
```

To sign into a personal Drive account interactively:

```bash
python scripts/upload_to_google_drive.py path/to/file-or-folder \
  --parent-folder-id "your-folder-id" \
  --oauth-client-secrets /secure/path/desktop-client.json
```

Use `--credentials-file path/to/service-account.json` only when ADC is not
already configured. Use `--impersonate-user user@example.org` only after a
Google Workspace administrator has configured domain-wide delegation.

Size buckets shown in the explorer use decoded size when available: Tiny below
100 MiB, Small below 1 GiB, Medium below 10 GiB, Large below 100 GiB, and Very
large at or above 100 GiB.
