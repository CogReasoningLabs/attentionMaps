# Google Drive uploads

`upload_to_drive.py` is a reusable uploader for a Google OAuth **Web
application** client. It does not start a desktop login server. The first
authorization happens in your web app; later uploads use the saved refresh
token without user interaction.

## One-time Google setup

1. In Google Cloud Console, select or create the project that will own the
   Drive integration, then enable the **Google Drive API**.
2. Configure **Google Auth Platform → Branding** and **Audience** first. If
   the app is in testing, add every account that will connect Drive as a test
   user (for example, `rijan4243@gmail.com`).
3. In **Google Auth Platform → Clients**, click **+ Create client**.
4. Choose the client type that matches the script you are using:

   | Use case | Client type | Script |
   | --- | --- | --- |
   | A one-user upload run on your own computer | **Desktop app** | `save.py` |
   | A deployed website/server that receives Google's callback | **Web application** | `upload_to_drive.py` |

5. Give the client a clear name and click **Create**. For a **Web
   application** client, add the callback URL under *Authorized redirect
   URIs* before creating it, for example:

   ```text
   http://localhost:8501/google/callback
   https://your-domain.com/google/callback
   ```

   Add only URLs your application actually handles. The URL passed by code
   must match one of these values exactly: scheme (`http`/`https`), domain,
   port, path, and trailing slash all matter.
6. In the **OAuth 2.0 Client IDs** list, click the client’s download icon, or
   select **Download JSON** in the “OAuth client created” dialog. Move the
   downloaded file into the application’s server-side folder and name it
   `credentials.json`.
7. Create/open the destination Drive folder. Its ID is the URL portion after
   `/folders/`.

`credentials.json` contains the client configuration. Do not commit it or put
it in a browser bundle. The generated `token.json` is more sensitive because
it grants Drive access; keep it on the server and out of Git as well.

### Important: do not mix Desktop and Web credentials

`save.py` calls `InstalledAppFlow.run_local_server(port=0)`, so it needs a
**Desktop app** client JSON. That flow opens a temporary local callback URL
with a random port.

`upload_to_drive.py` is for a **Web application** client. It expects your web
app to send the browser to Google and receive the redirect at a callback route
you registered in Cloud Console. Do not run `save.py` with a Web application
credential: it causes `Error 400: redirect_uri_mismatch`.

Install dependencies once:

```powershell
python -m pip install -r requirements.txt
```

For a one-user upload from your own computer, pass both the source path and
destination folder ID explicitly. `save.py` has no default upload target:

```powershell
python save.py upload_test_folder --folder-id YOUR_DRIVE_FOLDER_ID
```

## Add OAuth routes to your web app

Keep a random state in a signed server-side session; do not use a global
variable. Pass the entire callback URL (including `code` and `state`).

```python
import secrets
from upload_to_drive import get_authorization_url, save_credentials_from_callback

REDIRECT_URI = "https://your-domain.com/google/callback"

def start_google_connect(session):
    session["google_oauth_state"] = secrets.token_urlsafe(32)
    return get_authorization_url(REDIRECT_URI, session["google_oauth_state"])
    # Redirect the browser to the returned URL.

def google_callback(request_url, session):
    save_credentials_from_callback(
        request_url, REDIRECT_URI, session.pop("google_oauth_state")
    )
```

For a multi-user app, store each user's encrypted token JSON in a database or
secret store, rather than sharing the default `token.json`. Never return token
JSON to the browser.

## Upload from application code

```python
from upload_to_drive import upload_file, upload_text

folder_id = "YOUR_DRIVE_FOLDER_ID"
result = upload_file("data/processed/nepali_pretraining/train.parquet", folder_id=folder_id)
print(result["id"])

upload_text("model run complete\n", name="run-status.txt", folder_id=folder_id)
```

`upload_bytes()` uploads dynamically generated CSV, JSON, or other byte data.
The module requests the limited `drive.file` scope, which lets the app manage
files it creates.

After connecting once through the web app, a local CLI upload also works:

```powershell
python upload_to_drive.py data/processed/nepali_pretraining/train.parquet --folder-id YOUR_DRIVE_FOLDER_ID
```

Google requires an exact redirect-URI match. This module requests offline
access, so the server can refresh its access token later. See [Google's
web-server OAuth guide](https://developers.google.com/identity/protocols/oauth2/web-server).
