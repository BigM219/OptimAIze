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
    html = page.content()
    (OUT / 'history_dom.html').write_text(html, encoding='utf-8')
    info = page.evaluate("""
    () => Array.from(document.querySelectorAll('textarea,input,[id],label')).map((el) => ({
      tag: el.tagName,
      id: el.id,
      text: el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') || '',
      cls: el.className,
      disabled: el.disabled || false
    })).slice(0, 200)
    """)
    (OUT / 'history_dom_selectors.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(info)
    browser.close()
