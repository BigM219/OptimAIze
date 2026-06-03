import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path('outputs/ui/browser_tests')
OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    page.goto('http://127.0.0.1:7860', wait_until='networkidle', timeout=120000)
    page.get_by_role('tab', name='History', exact=True).click()
    page.wait_for_timeout(1000)
    info = page.evaluate("""
    () => Array.from(document.querySelectorAll('textarea,input')).map((el, idx) => ({
      idx,
      tag: el.tagName,
      value: el.value,
      disabled: el.disabled,
      aria: el.getAttribute('aria-label'),
      placeholder: el.getAttribute('placeholder'),
      rect: (() => { const r = el.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; })(),
      parentText: (el.closest('.block')?.innerText || el.parentElement?.innerText || '').slice(0, 200)
    }))
    """)
    (OUT / 'textareas.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=True))
    browser.close()
