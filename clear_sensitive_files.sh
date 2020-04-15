git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch settings.yaml" \
  --prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch credentials.json" \
  --prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch client_secrets.json" \
  --prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch utils/gdrive_utils.py" \
  --prune-empty --tag-name-filter cat -- --all
