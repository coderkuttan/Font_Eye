"""Proprietary fonts that can't be trained on (no open licence), keyed by the open
font the model knows that they closely resemble. Shown in the popup as a hint only,
never as the answer. Keep this to widely recognised resemblances.
"""
LOOKALIKES = {
    "Inter": "SF Pro (Apple)",
    "Montserrat": "Gotham",
    "Jost": "Futura",
    "League Spartan": "Futura",
    "Nunito Sans": "Avenir",
    "Helvetica": "Helvetica Neue",
    "Google Sans": "Product Sans",
    "DejaVu Sans Mono": "Menlo (macOS)",
}


def hint_for(name):
    """'Montserrat' or 'Helvetica / Nimbus Sans' -> 'Gotham' / 'Helvetica Neue' / None."""
    return LOOKALIKES.get(name) or LOOKALIKES.get(name.split(" / ")[0])
