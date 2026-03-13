"""
Targeted corpus expansion for T-1 (dry-run) and T2 (local destructive) tiers.

Current corpus: T-1=820 (2.9%), T2=1650 (5.8%)
Target:         T-1=2000+, T2=2500+

Strategy: systematic mutation of known patterns with realistic variation.

Usage:
    model/.venv/bin/python3 scripts/expand_corpus.py [--dry-run]
"""

import json
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CORPUS = Path("data/synthetic/scores-cache.jsonl")
DRY_RUN = "--dry-run" in sys.argv

# ---------------------------------------------------------------------------
# T-1: Dry-run / preview / check / test / validate patterns
# ---------------------------------------------------------------------------

# Template: (tool_prefix, dry_run_flag, args_variants)
# Each combination produces one entry.

DRY_RUN_FLAGS = {
    "kubectl": [
        ("--dry-run=client", [
            "apply -f deployment.yaml", "apply -f service.yaml", "apply -f configmap.yaml",
            "apply -f ingress.yaml", "apply -f cronjob.yaml", "apply -f daemonset.yaml",
            "apply -f statefulset.yaml", "apply -f networkpolicy.yaml",
            "create deployment nginx --image=nginx", "create service clusterip my-svc --tcp=80:8080",
            "create namespace test-ns", "create configmap my-config --from-literal=key=val",
            "delete pod test-pod", "delete deployment test-deploy",
            "set image deployment/app app=nginx:1.25", "set resources deployment/app -c=app --limits=cpu=200m",
            "label node worker-1 role=compute", "annotate pod test-pod description=test",
            "patch deployment app -p '{\"spec\":{\"replicas\":3}}'",
            "scale deployment/app --replicas=5", "autoscale deployment/app --min=2 --max=10",
            "rollout undo deployment/app", "rollout restart deployment/app",
        ]),
        ("--dry-run=server", [
            "apply -f deployment.yaml", "apply -f service.yaml",
            "create deployment test --image=nginx", "delete pod test-pod",
        ]),
    ],
    "helm": [
        ("--dry-run", [
            "install my-release ./chart", "install my-release bitnami/nginx",
            "install my-release ./chart -n staging", "install my-release ./chart --set replicaCount=3",
            "upgrade my-release ./chart", "upgrade my-release ./chart -n production",
            "upgrade --install my-release ./chart", "upgrade my-release bitnami/redis -n staging",
            "uninstall my-release", "uninstall my-release -n production",
            "rollback my-release 2", "rollback my-release 1 -n staging",
        ]),
    ],
    "terraform": [
        ("-lock=false", [  # terraform plan variants
            "plan", "plan -var-file=prod.tfvars", "plan -var-file=staging.tfvars",
            "plan -target=aws_instance.web", "plan -target=module.database",
            "plan -out=plan.tfplan", "plan -destroy",
        ]),
    ],
    "ansible-playbook": [
        ("--check", [
            "-i production site.yml", "-i staging deploy.yml", "-i inventory/hosts playbook.yml",
            "-i production deploy.yml --limit=webservers",
            "-i staging update.yml --tags=packages",
            "-i production rollback.yml --limit=db",
            "deploy.yml -i hosts --extra-vars='env=staging'",
            "provision.yml -i cloud.ini",
        ]),
        ("--check --diff", [
            "-i production site.yml", "-i staging deploy.yml",
            "-i production update.yml --limit=webservers",
            "deploy.yml -i hosts",
        ]),
    ],
    "rsync": [
        ("-avzn", [
            "./dist/ prod:/var/www/html/", "./build/ staging:/opt/app/",
            "--delete ./dist/ prod:/var/www/", "./src/ backup:/archive/src/",
            "-e ssh ./data/ remote:/backup/data/", "--exclude=node_modules ./app/ prod:/opt/",
            "--delete --exclude='.git' ./ prod:/deploy/",
            "./config/ prod:/etc/myapp/",
        ]),
        ("--dry-run", [
            "-avz ./dist/ prod:/var/www/html/", "-avz --delete ./build/ staging:/opt/",
            "-avz ./backup/ remote:/restore/",
        ]),
    ],
    "apt-get": [
        ("--simulate", [
            "install nginx", "install postgresql", "install redis-server",
            "install python3-pip", "install nodejs npm", "install docker.io",
            "remove nginx", "remove postgresql", "purge redis-server",
            "upgrade", "dist-upgrade", "autoremove",
        ]),
        ("-s", [
            "install nginx", "install python3", "upgrade",
            "remove apache2", "autoremove",
        ]),
    ],
    "dnf": [
        ("--assumeno", [
            "install nginx", "install postgresql", "install redis",
            "remove httpd", "update", "upgrade",
        ]),
    ],
    "yum": [
        ("--assumeno", [
            "install nginx", "install httpd", "update",
            "remove php", "upgrade",
        ]),
    ],
    "pip": [
        ("--dry-run", [
            "install requests", "install flask django", "install -r requirements.txt",
            "install --upgrade pip", "install numpy pandas",
            "uninstall requests", "uninstall flask",
        ]),
    ],
    "pip3": [
        ("--dry-run", [
            "install requests", "install flask django", "install -r requirements.txt",
            "install --upgrade pip", "install numpy pandas",
        ]),
    ],
    "docker": [
        ("--dry-run", [  # docker compose
            "compose up -d", "compose down", "compose build",
        ]),
    ],
    "make": [
        ("-n", [
            "deploy", "install", "clean", "build", "all",
            "release", "test", "package", "dist", "publish",
            "clean all", "deploy-staging", "deploy-production",
        ]),
        ("--dry-run", [
            "deploy", "install", "clean", "build",
            "release", "test",
        ]),
    ],
    "git": [
        ("--dry-run", [
            "push origin main", "push origin feature-branch", "push --force origin main",
            "push --tags", "clean -fd", "clean -fdx",
            "rm -r src/old/", "rm cached-file.txt",
            "add .", "add -A",
            "fetch origin", "fetch --all",
        ]),
    ],
    "rm": [
        ("-i", [  # interactive = confirmation before each delete
            "*.log", "/tmp/build/*", "node_modules",
            "-rf build/", "-rf dist/", "-rf .cache/",
        ]),
    ],
    "cp": [
        ("-i", [
            "src/config.json dst/config.json", "backup.sql /tmp/backup.sql",
        ]),
        ("-n", [  # no-clobber
            "src/config.json dst/config.json", "file.txt /tmp/file.txt",
        ]),
    ],
}

# Additional T-1 patterns: test/validate commands
TEST_COMMANDS = [
    # test frameworks
    "pytest --co", "pytest --collect-only", "pytest --co -q",
    "pytest tests/ --co", "pytest tests/unit/ --collect-only",
    "npm test -- --listTests", "npm test -- --listReporters",
    "jest --listTests", "jest --showConfig",
    "cargo test --no-run", "cargo test --no-run --release",
    "go test -list '.*' ./...", "go test -v -run TestNothing ./...",
    "gradle test --dry-run", "gradle build --dry-run",
    "mvn validate", "mvn verify -DskipTests",
    # linters / static analysis (preview mode)
    "eslint --fix-dry-run src/", "eslint --fix-dry-run src/**/*.ts",
    "eslint --fix-dry-run src/**/*.tsx",
    "prettier --check src/", "prettier --check .",
    "prettier --list-different src/", "prettier --list-different .",
    "black --check src/", "black --check .", "black --diff src/",
    "isort --check-only src/", "isort --diff src/",
    "rubocop --auto-correct-all --dry-run",
    "gofmt -d .", "gofmt -l .",
    "rustfmt --check src/main.rs", "rustfmt --check src/lib.rs",
    # terraform plan (always T-1)
    "terraform plan", "terraform plan -var-file=prod.tfvars",
    "terraform plan -out=plan.tfplan", "terraform plan -target=module.vpc",
    "terraform plan -destroy", "terraform plan -var='env=staging'",
    # ansible check
    "ansible all -m ping --check", "ansible webservers -m service -a 'name=nginx state=restarted' --check",
    # docker compose check
    "docker compose config", "docker compose config --services",
    "docker compose config --volumes",
    # airflow test (dry-run of a single task)
    "airflow tasks test my_dag my_task 2024-01-01",
    "airflow tasks test etl_pipeline extract_step 2024-03-01",
    "airflow dags test my_dag 2024-01-01",
    # npm/yarn check scripts
    "npm pack --dry-run", "npm publish --dry-run",
    "yarn pack --dry-run",
    # kubernetes diff
    "kubectl diff -f deployment.yaml", "kubectl diff -f service.yaml",
    "kubectl diff -f configmap.yaml -n production",
    # git diff/status/log (these are T0, but adding some mixed-flag variants as T-1)
    "git stash --dry-run", "git merge --no-commit --no-ff feature-branch",
    "git merge --no-commit develop", "git rebase --dry-run main",
    "git cherry-pick --no-commit abc123",
    # systemd analyze
    "systemd-analyze verify myservice.service",
    "systemd-analyze security myservice.service",
    # database check
    "pg_dump --schema-only mydb > /dev/null",  # schema check only
    "mysqlcheck --check mydb",
    "mysqlcheck --analyze mydb",
    # cloud preview
    "aws cloudformation validate-template --template-body file://template.yaml",
    "aws cloudformation create-change-set --stack-name prod --template-body file://template.yaml --change-set-name preview",
    "gcloud deployment-manager deployments update prod --preview",
    "gcloud deployment-manager deployments update staging --preview",
    "az deployment group what-if --resource-group prod --template-file main.bicep",
    "pulumi preview", "pulumi preview --stack production",
    "cdk diff", "cdk diff --all",
    "serverless deploy --noDeploy",
    "flyctl deploy --dry-run",
    # package manager simulation
    "brew install --dry-run nginx", "brew upgrade --dry-run",
    "apk add --simulate nginx", "apk del --simulate curl",
    "pacman -S --print nginx", "pacman -R --print httpd",
    "zypper install --dry-run nginx",
    "snap install --dry-run firefox",
    "flatpak install --dry-run flathub org.gimp.GIMP",
]

# Additional T-1: write tool with --no-op / preview flags
PREVIEW_COMMANDS = [
    "sed -n 's/old/new/gp' file.txt",  # sed print-only (no -i)
    "sed 's/foo/bar/g' config.yml",  # stdout only
    "awk '{gsub(/old/,\"new\")}1' file.txt",  # stdout transform
    "diff -u old_file.txt new_file.txt",
    "diff --color production.yaml staging.yaml",
    "diff3 base.txt mine.txt theirs.txt",
    "colordiff old.txt new.txt",
    "patch --dry-run < fix.patch",
    "patch --dry-run -p1 < security-fix.patch",
    "patch --check < update.patch",
    "dpkg --simulate -i package.deb",
    "rpm --test -i package.rpm",
    "rpm --test -U package.rpm",
]


# ---------------------------------------------------------------------------
# T2: Local destructive — deletes local data, modifies permissions,
# removes packages, resets state. NOT external/network mutations (T3)
# and NOT catastrophic (T4).
# ---------------------------------------------------------------------------

T2_COMMANDS = [
    # rm: local file deletion (non-catastrophic paths)
    "rm build/output.js", "rm -f dist/*.js", "rm -rf build/",
    "rm -rf dist/", "rm -rf .cache/", "rm -rf __pycache__/",
    "rm -rf coverage/", "rm -rf .pytest_cache/", "rm -rf .mypy_cache/",
    "rm -rf .next/", "rm -rf .nuxt/", "rm -rf .turbo/",
    "rm -rf target/debug/", "rm -rf target/release/",
    "rm -rf vendor/", "rm -f *.pyc", "rm -f *.o *.a",
    "rm -rf tmp/", "rm -rf /tmp/build-*", "rm -rf /tmp/test-*",
    "rm -f logs/*.log", "rm -f *.bak", "rm -f *.swp",
    "rm -rf .gradle/", "rm -rf .m2/repository/",
    "rm -f yarn.lock", "rm -f package-lock.json",
    "rm -f Gemfile.lock", "rm -f poetry.lock",
    "rm -f Cargo.lock", "rm -f go.sum",

    # git destructive (local-only)
    "git reset --hard HEAD~1", "git reset --hard HEAD~3",
    "git reset --hard HEAD", "git reset --hard origin/main",
    "git reset --mixed HEAD~1", "git reset --soft HEAD~1",
    "git clean -fd", "git clean -fdx", "git clean -f",
    "git checkout -- .", "git checkout -- src/",
    "git restore --staged .", "git restore .",
    "git stash drop", "git stash drop stash@{0}",
    "git stash clear", "git branch -D feature-branch",
    "git branch -D old-feature", "git branch -d merged-branch",
    "git rebase -i HEAD~5", "git rebase -i HEAD~10",
    "git rebase main", "git rebase develop",

    # docker local cleanup
    "docker rmi my-app:latest", "docker rmi nginx:1.24",
    "docker rmi -f $(docker images -q)", "docker rmi $(docker images --filter 'dangling=true' -q)",
    "docker rm container-name", "docker rm -f my-container",
    "docker volume rm my-data", "docker volume prune -f",
    "docker system prune -f", "docker system prune -af",
    "docker image prune -f", "docker image prune -af",
    "docker builder prune -f", "docker network prune -f",
    "docker container prune -f",
    "docker compose down", "docker compose down -v",
    "docker compose down --rmi all", "docker compose rm -sf",

    # chmod/chown (non-root paths)
    "chmod 644 src/config.json", "chmod 755 scripts/deploy.sh",
    "chmod +x scripts/*.sh", "chmod +x build.sh",
    "chmod -R 755 dist/", "chmod -R 644 public/",
    "chmod 600 .env", "chmod 600 ~/.ssh/config",
    "chown $(whoami) build/", "chown -R $(whoami) dist/",
    "chown www-data:www-data /var/www/html/", "chown -R www-data /var/www/",

    # package manager remove/uninstall (local)
    "pip uninstall requests", "pip uninstall flask django",
    "pip uninstall -y requests numpy pandas",
    "pip3 uninstall requests", "pip3 uninstall -y flask",
    "npm uninstall express", "npm uninstall lodash moment",
    "npm uninstall -g typescript", "npm uninstall -g eslint",
    "yarn remove express", "yarn remove lodash",
    "pnpm remove express", "pnpm remove -g typescript",
    "cargo remove serde", "cargo remove tokio",
    "gem uninstall rails", "gem uninstall bundler",
    "composer remove laravel/framework",
    "brew uninstall nginx", "brew uninstall redis",
    "brew remove postgresql", "brew cleanup",
    "brew cleanup --prune=all",
    "conda remove numpy", "conda remove pandas scipy",
    "conda env remove -n myenv",
    "pipenv uninstall requests", "pipenv --rm",
    "poetry remove requests", "poetry remove flask",

    # database local ops (T2: mutations that don't cross network boundaries for new infra)
    "sqlite3 app.db 'DELETE FROM sessions'",
    "sqlite3 app.db 'DROP TABLE temp_data'",
    "sqlite3 test.db 'VACUUM'",
    "redis-cli FLUSHDB", "redis-cli FLUSHALL",
    "redis-cli DEL session:expired:*",
    "redis-cli -n 1 FLUSHDB",

    # find -delete (non-system paths)
    "find . -name '*.pyc' -delete", "find . -name '__pycache__' -type d -exec rm -rf {} +",
    "find . -name '*.o' -delete", "find . -name '*.a' -delete",
    "find /tmp -name '*.log' -mtime +7 -delete",
    "find /tmp -name 'test-*' -mtime +1 -delete",
    "find . -name '.DS_Store' -delete",
    "find . -name 'Thumbs.db' -delete",
    "find build/ -type f -name '*.map' -delete",

    # truncate/overwrite
    "truncate -s 0 app.log", "truncate -s 0 /var/log/app.log",
    "> access.log", "> error.log",
    "echo '' > debug.log", "cat /dev/null > app.log",

    # make clean targets
    "make clean", "make distclean", "make clobber",
    "make mrproper", "make realclean",
    "gradle clean", "mvn clean",
    "cargo clean", "go clean",
    "rake clean", "rake clobber",

    # editor/IDE cleanup
    "rm -rf .idea/", "rm -rf .vscode/",
    "rm -rf *.iml", "rm -f .project .classpath",

    # python3 -m http.server (local server, mild risk)
    "python3 -m http.server 8080", "python3 -m http.server 3000",
    "python3 -m http.server 8000 --directory dist/",
    "php -S localhost:8000", "php -S 0.0.0.0:8080 -t public/",
    "ruby -run -e httpd . -p 3000",

    # tar overwrite (local file creation)
    "tar xzf archive.tar.gz", "tar xjf archive.tar.bz2",
    "tar xf backup.tar --overwrite",

    # sed in-place (local file modification)
    "sed -i 's/old/new/g' config.yml", "sed -i.bak 's/debug=true/debug=false/' .env",
    "sed -i 's/localhost/prod.internal/' config.json",
    "sed -i '/^#/d' config.conf",  # remove comments

    # mv (local renames/moves that could lose data)
    "mv old-dir/ archive/old-dir/", "mv config.json config.json.bak",
    "mv build/ build-old/", "mv dist/ dist-backup/",

    # kill processes
    "kill -9 12345", "kill -15 12345",
    "killall node", "killall python3",
    "pkill -f 'node server.js'", "pkill -f 'python3 app.py'",
]


def build_dry_run_entries() -> list[dict]:
    """Generate T-1 entries from dry-run flag templates."""
    entries = []
    for tool, flag_groups in DRY_RUN_FLAGS.items():
        for flag, args_list in flag_groups:
            for args in args_list:
                cmd = f"{tool} {args} {flag}" if tool in ("helm",) else f"{tool} {flag} {args}"
                # Some tools put flag before args, some after
                if tool in ("make",):
                    cmd = f"{tool} {flag} {args}"
                elif tool in ("git", "rm", "cp"):
                    cmd = f"{tool} {flag} {args}"
                elif tool in ("apt-get", "dnf", "yum", "pip", "pip3"):
                    cmd = f"{tool} {flag} {args}"
                    # Actually: apt-get --simulate install nginx
                    cmd = f"{tool} {flag} {args}"
                entries.append({
                    "text": cmd,
                    "risk": 0.12,
                    "tier": -1,
                    "scorer": "corpus-expansion",
                    "source": "dry-run-templates",
                })
    return entries


def build_test_entries() -> list[dict]:
    """Generate T-1 entries from test/validate commands."""
    entries = []
    for cmd in TEST_COMMANDS:
        entries.append({
            "text": cmd,
            "risk": 0.12,
            "tier": -1,
            "scorer": "corpus-expansion",
            "source": "test-validate-templates",
        })
    return entries


def build_preview_entries() -> list[dict]:
    """Generate T-1 entries from preview/check commands."""
    entries = []
    for cmd in PREVIEW_COMMANDS:
        entries.append({
            "text": cmd,
            "risk": 0.12,
            "tier": -1,
            "scorer": "corpus-expansion",
            "source": "preview-templates",
        })
    return entries


def build_t2_entries() -> list[dict]:
    """Generate T2 entries from local destructive commands."""
    entries = []
    for cmd in T2_COMMANDS:
        entries.append({
            "text": cmd,
            "risk": 0.42,
            "tier": 2,
            "scorer": "corpus-expansion",
            "source": "t2-destructive-templates",
        })
    return entries


def main():
    all_new = []
    dry_run = build_dry_run_entries()
    test = build_test_entries()
    preview = build_preview_entries()
    t2 = build_t2_entries()

    all_new.extend(dry_run)
    all_new.extend(test)
    all_new.extend(preview)
    all_new.extend(t2)

    print(f"Generated entries:")
    print(f"  T-1 dry-run flags: {len(dry_run)}")
    print(f"  T-1 test/validate: {len(test)}")
    print(f"  T-1 preview/check: {len(preview)}")
    print(f"  T2 local destruct: {len(t2)}")
    print(f"  Total: {len(all_new)}")

    # Deduplicate against existing corpus
    existing = set()
    if CORPUS.exists():
        for line in open(CORPUS):
            try:
                existing.add(json.loads(line).get("text", ""))
            except Exception:
                pass

    new_entries = [e for e in all_new if e["text"] not in existing]
    dupes = len(all_new) - len(new_entries)
    print(f"\n  Duplicates skipped: {dupes}")
    print(f"  New entries to add: {len(new_entries)}")

    tier_counts = Counter(e["tier"] for e in new_entries)
    print(f"\n  By tier:")
    for t in sorted(tier_counts):
        print(f"    T{t:+d}: {tier_counts[t]}")

    if DRY_RUN:
        print("\n--dry-run: no changes written")
        return

    ts = datetime.now(timezone.utc).isoformat()
    with open(CORPUS, "a") as f:
        for e in new_entries:
            e["id"] = str(uuid.uuid4())
            e["ts"] = ts
            f.write(json.dumps(e) + "\n")
    print(f"\nAppended {len(new_entries)} entries to {CORPUS}")

    # Show new totals
    all_entries = [json.loads(l) for l in open(CORPUS) if l.strip()]
    valid = [e for e in all_entries if e.get("tier") is not None]
    tiers = Counter(e["tier"] for e in valid)
    print(f"\nNew corpus totals ({len(valid):,} entries):")
    for t in sorted(tiers):
        pct = tiers[t] / len(valid) * 100
        print(f"  T{t:+d}: {tiers[t]:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
