#!/usr/bin/env python3
"""Generate T2/T3 boundary training data."""
import json

entries = [
    # ── T2: Local file/dir deletion (12) ──────────────────────────────────────
    {"command": "rm -rf ./build", "risk_score": 0.41, "blast_radius": 0.38, "tier": 2},
    {"command": "rm -rf ./dist", "risk_score": 0.40, "blast_radius": 0.37, "tier": 2},
    {"command": "rm -f *.log", "risk_score": 0.36, "blast_radius": 0.35, "tier": 2},
    {"command": "rm -rf /tmp/myapp-*", "risk_score": 0.42, "blast_radius": 0.39, "tier": 2},
    {"command": "find . -name '*.pyc' -delete", "risk_score": 0.37, "blast_radius": 0.35, "tier": 2},
    {"command": "find . -name '__pycache__' -type d -exec rm -rf {} +", "risk_score": 0.39, "blast_radius": 0.36, "tier": 2},
    {"command": "rm -rf node_modules/", "risk_score": 0.43, "blast_radius": 0.40, "tier": 2},
    {"command": "rm -rf .next/", "risk_score": 0.40, "blast_radius": 0.37, "tier": 2},
    {"command": "find /var/log/myapp -name '*.log' -mtime +7 -delete", "risk_score": 0.44, "blast_radius": 0.41, "tier": 2},
    {"command": "rm -rf ~/.cache/pip", "risk_score": 0.43, "blast_radius": 0.39, "tier": 2},
    {"command": "rm -rf target/", "risk_score": 0.41, "blast_radius": 0.38, "tier": 2},
    {"command": "rm -rf /opt/myapp/tmp/*", "risk_score": 0.46, "blast_radius": 0.43, "tier": 2},

    # ── T2: Git local state changes (12) ──────────────────────────────────────
    {"command": "git reset --hard HEAD~1", "risk_score": 0.44, "blast_radius": 0.42, "tier": 2},
    {"command": "git reset --hard HEAD~3", "risk_score": 0.48, "blast_radius": 0.46, "tier": 2},
    {"command": "git reset --hard origin/main", "risk_score": 0.50, "blast_radius": 0.48, "tier": 2},
    {"command": "git clean -fd", "risk_score": 0.43, "blast_radius": 0.41, "tier": 2},
    {"command": "git clean -fdx", "risk_score": 0.47, "blast_radius": 0.45, "tier": 2},
    {"command": "git stash drop stash@{0}", "risk_score": 0.41, "blast_radius": 0.39, "tier": 2},
    {"command": "git stash clear", "risk_score": 0.46, "blast_radius": 0.44, "tier": 2},
    {"command": "git branch -D feature/old-auth", "risk_score": 0.43, "blast_radius": 0.40, "tier": 2},
    {"command": "git tag -d v1.2.3", "risk_score": 0.40, "blast_radius": 0.38, "tier": 2},
    {"command": "git reflog expire --expire=now --all", "risk_score": 0.52, "blast_radius": 0.50, "tier": 2},
    {"command": "git checkout -- .", "risk_score": 0.46, "blast_radius": 0.44, "tier": 2},
    {"command": "git rebase -i HEAD~5", "risk_score": 0.48, "blast_radius": 0.45, "tier": 2},

    # ── T2: Local Docker ops (12) ─────────────────────────────────────────────
    {"command": "docker rmi my-app:latest", "risk_score": 0.40, "blast_radius": 0.38, "tier": 2},
    {"command": "docker rmi $(docker images -q)", "risk_score": 0.52, "blast_radius": 0.50, "tier": 2},
    {"command": "docker rm my-container", "risk_score": 0.38, "blast_radius": 0.36, "tier": 2},
    {"command": "docker rm $(docker ps -aq)", "risk_score": 0.50, "blast_radius": 0.48, "tier": 2},
    {"command": "docker volume rm myapp_data", "risk_score": 0.47, "blast_radius": 0.45, "tier": 2},
    {"command": "docker volume prune -f", "risk_score": 0.52, "blast_radius": 0.50, "tier": 2},
    {"command": "docker network rm myapp_network", "risk_score": 0.40, "blast_radius": 0.38, "tier": 2},
    {"command": "docker system prune -af", "risk_score": 0.54, "blast_radius": 0.52, "tier": 2},
    {"command": "docker stop my-container", "risk_score": 0.37, "blast_radius": 0.36, "tier": 2},
    {"command": "docker compose down -v", "risk_score": 0.50, "blast_radius": 0.48, "tier": 2},
    {"command": "docker rmi myregistry.local/myapp:staging", "risk_score": 0.42, "blast_radius": 0.40, "tier": 2},
    {"command": "docker compose rm -f", "risk_score": 0.44, "blast_radius": 0.42, "tier": 2},

    # ── T2: Local package removal (12) ───────────────────────────────────────
    {"command": "pip uninstall flask -y", "risk_score": 0.39, "blast_radius": 0.37, "tier": 2},
    {"command": "pip uninstall -r requirements.txt -y", "risk_score": 0.50, "blast_radius": 0.48, "tier": 2},
    {"command": "npm uninstall express", "risk_score": 0.38, "blast_radius": 0.36, "tier": 2},
    {"command": "npm uninstall -g create-react-app", "risk_score": 0.42, "blast_radius": 0.40, "tier": 2},
    {"command": "brew uninstall postgresql", "risk_score": 0.44, "blast_radius": 0.42, "tier": 2},
    {"command": "brew uninstall --force node", "risk_score": 0.47, "blast_radius": 0.45, "tier": 2},
    {"command": "apt-get remove --purge nginx -y", "risk_score": 0.48, "blast_radius": 0.46, "tier": 2},
    {"command": "apt-get autoremove -y", "risk_score": 0.45, "blast_radius": 0.43, "tier": 2},
    {"command": "cargo clean", "risk_score": 0.38, "blast_radius": 0.36, "tier": 2},
    {"command": "gem uninstall rails --all", "risk_score": 0.43, "blast_radius": 0.41, "tier": 2},
    {"command": "conda remove --name myenv --all", "risk_score": 0.49, "blast_radius": 0.47, "tier": 2},
    {"command": "pyenv uninstall 3.9.0", "risk_score": 0.42, "blast_radius": 0.40, "tier": 2},

    # ── T2: Local database ops (12) ──────────────────────────────────────────
    {"command": "dropdb myapp_dev", "risk_score": 0.51, "blast_radius": 0.49, "tier": 2},
    {"command": "dropdb --if-exists myapp_test", "risk_score": 0.47, "blast_radius": 0.45, "tier": 2},
    {"command": "sqlite3 app.db 'DROP TABLE sessions'", "risk_score": 0.48, "blast_radius": 0.46, "tier": 2},
    {"command": "sqlite3 app.db 'DELETE FROM users WHERE created_at < date(\"now\",\"-30 days\")'", "risk_score": 0.50, "blast_radius": 0.48, "tier": 2},
    {"command": "psql -U postgres -c 'TRUNCATE users CASCADE' myapp_dev", "risk_score": 0.52, "blast_radius": 0.50, "tier": 2},
    {"command": "psql -U postgres -c 'DROP DATABASE myapp_dev'", "risk_score": 0.53, "blast_radius": 0.51, "tier": 2},
    {"command": "mysql -u root -proot myapp_dev -e 'DROP TABLE orders'", "risk_score": 0.51, "blast_radius": 0.49, "tier": 2},
    {"command": "redis-cli FLUSHDB", "risk_score": 0.48, "blast_radius": 0.46, "tier": 2},
    {"command": "redis-cli FLUSHALL", "risk_score": 0.54, "blast_radius": 0.52, "tier": 2},
    {"command": "mongo myapp_dev --eval 'db.dropDatabase()'", "risk_score": 0.53, "blast_radius": 0.51, "tier": 2},
    {"command": "pg_restore --clean -d myapp_dev backup.dump", "risk_score": 0.50, "blast_radius": 0.48, "tier": 2},
    {"command": "mysql -u root -e 'DROP DATABASE myapp_test'", "risk_score": 0.51, "blast_radius": 0.49, "tier": 2},

    # ── T2: Permission/ownership changes (10) ────────────────────────────────
    {"command": "chmod -R 777 ./uploads", "risk_score": 0.46, "blast_radius": 0.44, "tier": 2},
    {"command": "chmod -R 600 ~/.ssh", "risk_score": 0.43, "blast_radius": 0.41, "tier": 2},
    {"command": "chmod -R 755 /var/www/html", "risk_score": 0.45, "blast_radius": 0.43, "tier": 2},
    {"command": "chown -R www-data:www-data /var/www", "risk_score": 0.47, "blast_radius": 0.45, "tier": 2},
    {"command": "chown -R deploy:deploy /opt/myapp", "risk_score": 0.44, "blast_radius": 0.42, "tier": 2},
    {"command": "chgrp -R docker /var/run/docker.sock", "risk_score": 0.48, "blast_radius": 0.46, "tier": 2},
    {"command": "chmod -R g+w /shared/data", "risk_score": 0.46, "blast_radius": 0.44, "tier": 2},
    {"command": "chmod 4755 /usr/local/bin/myapp-helper", "risk_score": 0.50, "blast_radius": 0.48, "tier": 2},
    {"command": "chmod 000 /etc/cron.d/myapp", "risk_score": 0.49, "blast_radius": 0.47, "tier": 2},
    {"command": "chown root:root /etc/sudoers.d/myapp", "risk_score": 0.52, "blast_radius": 0.50, "tier": 2},

    # ── T2: Config file modification (10) ────────────────────────────────────
    {"command": "sed -i 's/DEBUG=True/DEBUG=False/' .env", "risk_score": 0.44, "blast_radius": 0.42, "tier": 2},
    {"command": "sed -i '/^#/d' /etc/nginx/nginx.conf", "risk_score": 0.48, "blast_radius": 0.46, "tier": 2},
    {"command": "truncate -s 0 /var/log/myapp/app.log", "risk_score": 0.41, "blast_radius": 0.39, "tier": 2},
    {"command": "> /etc/hosts", "risk_score": 0.54, "blast_radius": 0.52, "tier": 2},
    {"command": "cp /dev/null /etc/cron.d/myapp", "risk_score": 0.50, "blast_radius": 0.48, "tier": 2},
    {"command": "sed -i 's|127.0.0.1|0.0.0.0|g' /etc/redis/redis.conf", "risk_score": 0.52, "blast_radius": 0.50, "tier": 2},
    {"command": "echo 'vm.overcommit_memory=1' >> /etc/sysctl.conf && sysctl -p", "risk_score": 0.51, "blast_radius": 0.49, "tier": 2},
    {"command": "crontab -r", "risk_score": 0.49, "blast_radius": 0.47, "tier": 2},
    {"command": "systemctl disable myapp.service", "risk_score": 0.46, "blast_radius": 0.44, "tier": 2},
    {"command": "iptables -F", "risk_score": 0.54, "blast_radius": 0.53, "tier": 2},

    # ── T3: Remote git ops (14) ───────────────────────────────────────────────
    {"command": "git push origin main", "risk_score": 0.57, "blast_radius": 0.58, "tier": 3},
    {"command": "git push --force-with-lease origin main", "risk_score": 0.68, "blast_radius": 0.72, "tier": 3},
    {"command": "git push --force origin main", "risk_score": 0.74, "blast_radius": 0.76, "tier": 3},
    {"command": "git push origin --delete feature/old-branch", "risk_score": 0.62, "blast_radius": 0.63, "tier": 3},
    {"command": "git push --tags", "risk_score": 0.60, "blast_radius": 0.61, "tier": 3},
    {"command": "git push origin v2.0.0", "risk_score": 0.61, "blast_radius": 0.62, "tier": 3},
    {"command": "git push --mirror backup-remote", "risk_score": 0.70, "blast_radius": 0.73, "tier": 3},
    {"command": "git push heroku main", "risk_score": 0.72, "blast_radius": 0.74, "tier": 3},
    {"command": "git push --all origin", "risk_score": 0.65, "blast_radius": 0.66, "tier": 3},
    {"command": "git push origin --delete v0.9.0", "risk_score": 0.63, "blast_radius": 0.64, "tier": 3},
    {"command": "git push --force-with-lease origin release/2.0", "risk_score": 0.70, "blast_radius": 0.73, "tier": 3},
    {"command": "git push upstream HEAD:refs/for/main", "risk_score": 0.60, "blast_radius": 0.61, "tier": 3},
    {"command": "git subtree push --prefix=dist origin gh-pages", "risk_score": 0.64, "blast_radius": 0.65, "tier": 3},
    {"command": "git push --force origin --tags", "risk_score": 0.72, "blast_radius": 0.75, "tier": 3},

    # ── T3: Container registry (12) ──────────────────────────────────────────
    {"command": "docker push myregistry.io/myapp:latest", "risk_score": 0.60, "blast_radius": 0.62, "tier": 3},
    {"command": "docker push myregistry.io/myapp:v2.0.0", "risk_score": 0.61, "blast_radius": 0.63, "tier": 3},
    {"command": "docker buildx build --push -t myregistry.io/myapp:latest .", "risk_score": 0.63, "blast_radius": 0.65, "tier": 3},
    {"command": "docker buildx build --push -t myregistry.io/myapp:latest --platform linux/amd64,linux/arm64 .", "risk_score": 0.66, "blast_radius": 0.68, "tier": 3},
    {"command": "skopeo copy docker://myapp:latest docker://registry.io/myapp:latest", "risk_score": 0.62, "blast_radius": 0.64, "tier": 3},
    {"command": "crane push myapp.tar gcr.io/my-project/myapp:latest", "risk_score": 0.63, "blast_radius": 0.65, "tier": 3},
    {"command": "docker push gcr.io/my-project/api:$(git rev-parse --short HEAD)", "risk_score": 0.62, "blast_radius": 0.64, "tier": 3},
    {"command": "docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/myapp:latest", "risk_score": 0.63, "blast_radius": 0.65, "tier": 3},
    {"command": "aws ecr batch-delete-image --repository-name myapp --image-ids imageTag=old", "risk_score": 0.67, "blast_radius": 0.68, "tier": 3},
    {"command": "gcloud container images delete gcr.io/my-project/myapp:v1.0 --force-delete-tags", "risk_score": 0.69, "blast_radius": 0.70, "tier": 3},
    {"command": "crane tag gcr.io/my-project/myapp:sha-abc123 latest", "risk_score": 0.59, "blast_radius": 0.60, "tier": 3},
    {"command": "docker push quay.io/myorg/myapp:production", "risk_score": 0.62, "blast_radius": 0.64, "tier": 3},

    # ── T3: Package publishing (12) ───────────────────────────────────────────
    {"command": "npm publish", "risk_score": 0.64, "blast_radius": 0.67, "tier": 3},
    {"command": "npm publish --tag latest", "risk_score": 0.66, "blast_radius": 0.69, "tier": 3},
    {"command": "npm unpublish my-package@1.0.0", "risk_score": 0.72, "blast_radius": 0.74, "tier": 3},
    {"command": "pip publish dist/*", "risk_score": 0.63, "blast_radius": 0.66, "tier": 3},
    {"command": "twine upload dist/*", "risk_score": 0.63, "blast_radius": 0.66, "tier": 3},
    {"command": "twine upload --repository pypi dist/*", "risk_score": 0.65, "blast_radius": 0.68, "tier": 3},
    {"command": "cargo publish", "risk_score": 0.63, "blast_radius": 0.66, "tier": 3},
    {"command": "cargo publish --token $CRATES_IO_TOKEN", "risk_score": 0.65, "blast_radius": 0.68, "tier": 3},
    {"command": "gem push my-gem-1.0.0.gem", "risk_score": 0.63, "blast_radius": 0.66, "tier": 3},
    {"command": "gem yank my-gem --version 0.9.0", "risk_score": 0.70, "blast_radius": 0.72, "tier": 3},
    {"command": "helm push mychart-1.0.0.tgz oci://registry.io/charts", "risk_score": 0.64, "blast_radius": 0.66, "tier": 3},
    {"command": "mvn deploy -DskipTests", "risk_score": 0.65, "blast_radius": 0.67, "tier": 3},

    # ── T3: Cloud storage (14) ───────────────────────────────────────────────
    {"command": "aws s3 cp ./dist s3://my-bucket/dist --recursive", "risk_score": 0.62, "blast_radius": 0.64, "tier": 3},
    {"command": "aws s3 sync ./dist s3://my-bucket/", "risk_score": 0.63, "blast_radius": 0.65, "tier": 3},
    {"command": "aws s3 rm s3://my-bucket/dist --recursive", "risk_score": 0.68, "blast_radius": 0.70, "tier": 3},
    {"command": "aws s3 rm s3://prod-assets/ --recursive", "risk_score": 0.74, "blast_radius": 0.77, "tier": 3},
    {"command": "gsutil cp -r ./dist gs://my-bucket/", "risk_score": 0.62, "blast_radius": 0.64, "tier": 3},
    {"command": "gsutil rsync -r -d ./dist gs://my-bucket/dist", "risk_score": 0.66, "blast_radius": 0.68, "tier": 3},
    {"command": "gsutil rm -r gs://my-bucket/old-release/", "risk_score": 0.68, "blast_radius": 0.70, "tier": 3},
    {"command": "az storage blob upload-batch -s ./dist -d '$web' --account-name mystorageacct", "risk_score": 0.63, "blast_radius": 0.65, "tier": 3},
    {"command": "az storage blob delete-batch -s '$web' --account-name mystorageacct", "risk_score": 0.70, "blast_radius": 0.72, "tier": 3},
    {"command": "aws s3 cp ./backup.sql.gz s3://backups-bucket/$(date +%Y%m%d)/", "risk_score": 0.60, "blast_radius": 0.62, "tier": 3},
    {"command": "rclone sync ./dist remote:my-bucket --delete-excluded", "risk_score": 0.67, "blast_radius": 0.69, "tier": 3},
    {"command": "aws s3 rb s3://old-staging-bucket --force", "risk_score": 0.74, "blast_radius": 0.76, "tier": 3},
    {"command": "gsutil -m rm -r gs://staging-assets/v1/", "risk_score": 0.67, "blast_radius": 0.69, "tier": 3},
    {"command": "aws s3 sync . s3://prod-config-bucket/ --delete", "risk_score": 0.72, "blast_radius": 0.75, "tier": 3},

    # ── T3: K8s cluster ops (16) ─────────────────────────────────────────────
    {"command": "kubectl apply -f deployment.yaml", "risk_score": 0.62, "blast_radius": 0.64, "tier": 3},
    {"command": "kubectl apply -f k8s/", "risk_score": 0.65, "blast_radius": 0.67, "tier": 3},
    {"command": "kubectl delete pod my-pod", "risk_score": 0.60, "blast_radius": 0.62, "tier": 3},
    {"command": "kubectl delete deployment my-app", "risk_score": 0.68, "blast_radius": 0.70, "tier": 3},
    {"command": "kubectl delete namespace staging", "risk_score": 0.74, "blast_radius": 0.77, "tier": 3},
    {"command": "kubectl scale deployment my-app --replicas=0", "risk_score": 0.66, "blast_radius": 0.68, "tier": 3},
    {"command": "kubectl rollout restart deployment/my-app", "risk_score": 0.62, "blast_radius": 0.64, "tier": 3},
    {"command": "kubectl rollout undo deployment/my-app", "risk_score": 0.64, "blast_radius": 0.66, "tier": 3},
    {"command": "helm upgrade my-release ./chart --values prod-values.yaml", "risk_score": 0.68, "blast_radius": 0.70, "tier": 3},
    {"command": "helm install my-release ./chart -n production", "risk_score": 0.66, "blast_radius": 0.68, "tier": 3},
    {"command": "helm uninstall my-release -n production", "risk_score": 0.70, "blast_radius": 0.72, "tier": 3},
    {"command": "kubectl delete service my-svc", "risk_score": 0.65, "blast_radius": 0.67, "tier": 3},
    {"command": "kubectl apply -f https://raw.githubusercontent.com/org/repo/main/k8s/prod.yaml", "risk_score": 0.72, "blast_radius": 0.74, "tier": 3},
    {"command": "kubectl drain node-01 --ignore-daemonsets --delete-emptydir-data", "risk_score": 0.70, "blast_radius": 0.73, "tier": 3},
    {"command": "kubectl cordon node-02", "risk_score": 0.61, "blast_radius": 0.63, "tier": 3},
    {"command": "kubectl delete pvc my-app-data", "risk_score": 0.72, "blast_radius": 0.74, "tier": 3},

    # ── T3: API/webhook calls (12) ────────────────────────────────────────────
    {"command": "curl -X POST https://api.prod.internal/deploy -H 'Authorization: Bearer $TOKEN' -d '{\"version\":\"2.0\"}'", "risk_score": 0.70, "blast_radius": 0.73, "tier": 3},
    {"command": "curl -X DELETE https://api.example.com/v1/users/123 -H 'Authorization: Bearer $TOKEN'", "risk_score": 0.68, "blast_radius": 0.70, "tier": 3},
    {"command": "curl -X POST https://hooks.slack.com/services/T00/B00/xxx -d '{\"text\":\"Deploy started\"}'", "risk_score": 0.57, "blast_radius": 0.58, "tier": 3},
    {"command": "curl -X POST https://api.pagerduty.com/incidents -H 'Authorization: Token token=$PD_TOKEN' -d @incident.json", "risk_score": 0.66, "blast_radius": 0.68, "tier": 3},
    {"command": "curl -X POST https://api.stripe.com/v1/refunds -u $STRIPE_KEY: -d charge=ch_xxx", "risk_score": 0.72, "blast_radius": 0.75, "tier": 3},
    {"command": "curl -X POST https://smtp.mailgun.org/v3/mydomain.com/messages -F to='all@mydomain.com' -F text='Deploy complete'", "risk_score": 0.65, "blast_radius": 0.67, "tier": 3},
    {"command": "curl -X POST https://api.github.com/repos/org/repo/releases -H 'Authorization: token $GH_TOKEN' -d @release.json", "risk_score": 0.64, "blast_radius": 0.66, "tier": 3},
    {"command": "curl -X DELETE https://api.cloudflare.com/client/v4/zones/$ZONE_ID/purge_cache -d '{\"purge_everything\":true}'", "risk_score": 0.67, "blast_radius": 0.69, "tier": 3},
    {"command": "curl -X POST https://app.terraform.io/api/v2/runs -H 'Authorization: Bearer $TF_TOKEN' -d @run.json", "risk_score": 0.72, "blast_radius": 0.74, "tier": 3},
    {"command": "curl -X PUT https://api.prod.company.com/config -H 'Content-Type: application/json' -d @config.json", "risk_score": 0.70, "blast_radius": 0.72, "tier": 3},
    {"command": "aws lambda invoke --function-name prod-data-cleanup --payload '{}' /dev/null", "risk_score": 0.69, "blast_radius": 0.71, "tier": 3},
    {"command": "aws sns publish --topic-arn arn:aws:sns:us-east-1:123:prod-alerts --message 'Deploying v2.0'", "risk_score": 0.59, "blast_radius": 0.60, "tier": 3},
]

assert len(entries) == 160, f"Expected 160, got {len(entries)}"
t2 = [e for e in entries if e["tier"] == 2]
t3 = [e for e in entries if e["tier"] == 3]
assert len(t2) == 80, f"Expected 80 T2, got {len(t2)}"
assert len(t3) == 80, f"Expected 80 T3, got {len(t3)}"

print(json.dumps(entries, indent=2))
