# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly by emailing [your-email@example.com]. Do not open a public GitHub issue for security vulnerabilities.

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ Current |

## Security Practices

- **Secrets Management**: All API keys and credentials are stored in environment variables, never committed to source control.
- **Dependencies**: Regularly audited via `pip audit` and Dependabot.
- **CORS**: Restricted to known origins in production.
- **Input Validation**: All API inputs validated via Pydantic schemas.
- **No PII**: The system processes only anonymous joint angle data—no images, video, or personal identifiers are stored.

## Azure Security

- Azure OpenAI accessed via API key (MVP) with a path to Managed Identity for production.
- App Service configured with HTTPS-only and minimum TLS 1.2.
