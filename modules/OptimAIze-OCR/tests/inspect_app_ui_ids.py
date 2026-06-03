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
    () => Array.from(document.querySelectorAll('[id]')).map((el) => ({
      tag: el.tagName,
      id: el.id,
      text: (el.innerText || el.value || '').slice(0, 120),
      html: el.outerHTML.slice(0, 300)
    }))
    """)
    (OUT / 'ids.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps([x for x in info if 'history' in x['id'].lower()], ensure_ascii=True))
    browser.close()
