---
description: At the start of every session, always check the .ai-rules.md file in the project root and follow its guidelines strictly.
---

# Project ChatFlow - AI Agent Guidelines

If you are an AI assistant (like Antigravity) working on this repository, you **MUST** follow these rules to ensure consistency and speed across sessions.

## 1. Environment & Tools
*   **Virtual Environment**: ALWAYS ensure the virtual environment is activated before running any Django or Python commands (e.g., `venv\Scripts\activate` on Windows).
*   **Planning Mode**: Never modify source code without first researching the problem and providing an `implementation_plan.md` for user approval.
*   **Task Management**: Track your progress in `task.md` and summarize results in `walkthrough.md`.

## 2. Design Standards (The "ChatFlow" Theme)
The application MUST strictly follow these aesthetic guidelines:
*   **Background**: Use `#18181b` (Zinc-900) or `#111113` for main containers.
*   **Accents**: Use `#8B5CF6` (Violet-500) for buttons, active states, and focus borders.
*   **Atmosphere**: Use Glassmorphism (backdrop-blur, subtle borders, translucent overlays).
*   **Typography**: Prioritize "DM Sans" for general text and "JetBrains Mono" for code or data.
*   **Rich UI**: Use `lucide` icons and `apexcharts` for data visualization.

## 3. Technology Stack
*   **Interactivity**: Favor **HTMX** (or lightweight `fetch` JS) over full page reloads.
*   **Templating**: Use semantic HTML5 and Django's template system.
*   **CSS**: Stick to high-quality Vanilla CSS. Avoid Tailwind unless explicitly requested.
*   **Authentication**: Use `django-allauth` for standard and social login. Secure all internal views with `@login_required`.

## 4. Communication Style
*   Be concise and professional.
*   Always provide clickable file links when referencing project files.
*   Never use placeholders; always use the `generate_image` tool or real assets.

---
**Note to Agent**: If you've read this, acknowledge it briefly at the start of your turn to let the USER know you're in sync with the project's standards.
