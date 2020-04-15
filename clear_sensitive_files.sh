git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch settings.yaml" \
  --prune-empty --tag-name-filter cat -- --all
