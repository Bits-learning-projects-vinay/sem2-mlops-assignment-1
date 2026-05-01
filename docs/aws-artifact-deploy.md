# AWS model artifact deployment

This repository uploads trained files from `artifacts/` to Amazon S3 during CI.

## 1) Rotate exposed credentials first

If an access key was shared in chat, logs, or source control, rotate it immediately in AWS IAM.

## 2) Create required GitHub repository secrets

In **Settings -> Secrets and variables -> Actions -> New repository secret**, add:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (example: `ap-south-1`)
- `S3_BUCKET` (bucket name only, no `s3://`)
- `S3_PREFIX` (optional, example: `models/latest`)

## 3) Minimal IAM permissions

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
3. If all AWS secrets are present on `push` events, runs `aws s3 sync artifacts/ ... --delete`.

If AWS secrets are missing, the S3 upload step is skipped and the pipeline still runs.

