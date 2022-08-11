# pull remote changes, merge them and reset master branch to the remote changes.
git fetch --all
git reset --hard origin/main
