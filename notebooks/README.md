# Notebook policy

Notebooks are optional exploratory surfaces, not the source of truth for this project.
All benchmark logic lives in `src/`, and every committed result is generated through
the package CLI. If an exploratory notebook is added later, it should import the
package rather than duplicate experiment code.
