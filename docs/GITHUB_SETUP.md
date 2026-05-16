# GitHub Setup

## Create Repository

Repository name:

```text
calligraphy
```

Recommended visibility while collaborating:

```text
Private
```

After creating the empty GitHub repository, connect local repo:

```bash
cd /Users/admin/Desktop/calligraphy_generation_algo
git remote add origin git@github.com:<owner>/calligraphy.git
git branch -M main
git push -u origin main
```

If using HTTPS:

```bash
git remote add origin https://github.com/<owner>/calligraphy.git
git branch -M main
git push -u origin main
```

## Invite Collaborator

GitHub repository page:

```text
Settings -> Collaborators -> Add people
```

## Protect Main

Recommended:

- Require pull request before merging.
- Require at least one approval when possible.
- Do not commit datasets or checkpoints.
- Use Issues for training monitor logs.

