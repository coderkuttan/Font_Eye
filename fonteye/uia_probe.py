"""Ask Windows UI Automation which font an app used at a screen point.

When the app exposes its text (Word, Edge/Chrome pages, Notepad, VS Code, most
native apps) this is the *exact* font, not a guess.
"""
import uiautomation as auto

WEIGHTS = {100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular", 500: "Medium",
           600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black"}


def _clean(v):
    # UIA returns a sentinel object when a range has mixed values
    return v if isinstance(v, (str, int, float)) and v not in ("", 0) else None


def font_at(x, y, max_parents=6):
    """Return {'name','size','weight','italic','app'} or None. x, y: physical pixels."""
    try:
        with auto.UIAutomationInitializerInThread():
            ctrl = auto.ControlFromPoint(x, y)
            app = None
            for _ in range(max_parents):
                if ctrl is None:
                    break
                tp = ctrl.GetPattern(auto.PatternId.TextPattern)
                if tp is not None:
                    rng = tp.RangeFromPoint(x, y)
                    if rng is None:
                        break
                    rng.ExpandToEnclosingUnit(auto.TextUnit.Word)
                    A = auto.TextAttributeId
                    name = _clean(rng.GetAttributeValue(A.FontNameAttribute))
                    if not name:
                        break
                    size = _clean(rng.GetAttributeValue(A.FontSizeAttribute))
                    weight = _clean(rng.GetAttributeValue(A.FontWeightAttribute))
                    italic = rng.GetAttributeValue(A.IsItalicAttribute) is True
                    try:
                        app = ctrl.GetTopLevelControl().Name
                    except Exception:  # noqa: BLE001
                        pass
                    return {"name": name, "size": size,
                            "weight": WEIGHTS.get(int(weight), str(weight)) if weight else None,
                            "italic": italic, "text": rng.GetText(40).strip(), "app": app}
                ctrl = ctrl.GetParentControl()
    except Exception:  # noqa: BLE001
        return None
    return None


if __name__ == "__main__":
    import time
    print("Hover over some text... reading in 3 s")
    time.sleep(3)
    print(font_at(*auto.GetCursorPos()))
