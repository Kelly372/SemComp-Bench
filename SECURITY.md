# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub's private vulnerability
reporting or security-advisory feature when it is enabled for this repository.
If it is unavailable, contact the repository maintainers privately before
opening a public issue.

Do not include active credentials, private media, personal data, or an
unredacted provider response in a report. Include the affected file, impact,
reproduction steps using synthetic data, and a suggested mitigation when
possible.

## Credential handling

The pipeline reads service credentials from `VLM_API_KEY`. Never commit a real
key to the repository, a configuration file, a log, or a test fixture. If a
credential is exposed, revoke or rotate it with the provider immediately;
removing it from the latest Git commit is not sufficient.

Generated media and intermediate Parquet/JSON files may contain source URLs,
provider response IDs, or other sensitive metadata. They are ignored by
default and should be reviewed before sharing.
