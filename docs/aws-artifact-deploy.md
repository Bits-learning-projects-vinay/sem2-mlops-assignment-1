# AWS model artifact deployment

This repository uploads trained files from `artifacts/` to Amazon S3 during CI.

## 1) Rotate exposed credentials first

If an access key was shared in chat, logs, or source control, rotate it immediately in AWS IAM.

## 2) Create required GitHub repository secrets

In **Settings -> Secrets and variables -> Actions -> New repository secret**, add:

- `AWS_ROLE_ARN` (example: `arn:aws:iam::824122062678:role/GitHubActionsModelUploadRole`)
- `AWS_REGION` (example: `ap-south-1`)
- `S3_BUCKET` (bucket name only, no `s3://`)
- `S3_PREFIX` (optional, example: `models/latest`)

## 3) Configure IAM role and permissions

Create an IAM role trusted by GitHub OIDC provider (`token.actions.githubusercontent.com`) for your repository/branch.
Use `AWS_ROLE_ARN` in GitHub secrets instead of long-lived access keys.

Trust policy example:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::824122062678:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:<ORG_OR_USER>/<REPO>:ref:refs/heads/*"
        }
      }
    }
  ]
}
```


Attach least-privilege policy for your target bucket/prefix. Example:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::YOUR_BUCKET"
    },
    {
      "Effect": "Allow",
      "Action": [
          "s3:PutObject",
          "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::YOUR_BUCKET/YOUR_PREFIX/*"
    }
  ]
}
```

## 4) How it works

The CI workflow in `.github/workflows/ci.yml` now:

1. Trains the model.
2. Uploads local artifacts to GitHub Actions artifacts.
3. If `AWS_ROLE_ARN`, `AWS_REGION`, and `S3_BUCKET` are present on `push`, assumes role with OIDC and runs `aws s3 sync artifacts/ ... --delete`.

If AWS secrets are missing, the S3 upload step is skipped and the pipeline still runs.

