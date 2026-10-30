"""Chromium smoke test for the bidirectional timeline. Optional local test, not needed on Streamlit Cloud."""
from pathlib import Path
from playwright.sync_api import sync_playwright

page_path=Path(__file__).resolve().parents[1]/'timeline_component'/'index.html'
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
    page=browser.new_page(viewport={'width':1280,'height':1100},device_scale_factor=1)
    page.set_content(page_path.read_text(encoding="utf-8"))
    page.evaluate("""() => { window._testMessages=[]; window.postMessage = (data) => window._testMessages.push(data); }""")
    page.evaluate("""() => window.dispatchEvent(new MessageEvent('message',{data:{type:'streamlit:render',args:{duration:21.69,signature:'video-check1',thumbs:[],segments:[],video_b64:'',mime:'video/mp4'}}}))""")
    assert '0.00 / 21.69 秒' in page.locator('#timeDisplay').inner_text()
    page.locator('#add').click()
    assert page.locator('.segment').count()==1
    assert page.locator('.handle.left').count()==1
    assert page.locator('.handle.right').count()==1
    page.locator('#title').fill('拿起箱子')
    page.locator('#title').press('Tab')
    page.locator('#muscle').select_option('1')
    page.locator('#force').select_option('2')
    page.locator('#start').fill('4'); page.locator('#start').press('Tab')
    page.locator('#end').fill('9'); page.locator('#end').press('Tab')
    # Add a second custom segment by two clicks at 12s and 17s.
    page.locator('#addByClick').click()
    strip=page.locator('#strip')
    box=strip.bounding_box()
    for t in [12,17]:
        page.mouse.click(box['x']+box['width']*(t/21.69),box['y']+box['height']/2)
    assert page.locator('.segment').count()==2
    page.locator('#force').select_option('3')
    # drag right endpoint of second interval from 17 to 18s
    right=page.locator('.segment.selected .handle.right')
    rbox=right.bounding_box()
    page.mouse.move(rbox['x']+rbox['width']/2,rbox['y']+rbox['height']/2)
    page.mouse.down()
    page.mouse.move(rbox['x']+rbox['width']/2+box['width']*(1/21.69),rbox['y']+rbox['height']/2,steps=5)
    page.mouse.up()
    assert float(page.locator('#end').input_value())>17.4
    page.locator('#save').click()
    msg=page.evaluate("() => window._testMessages.filter(x => x.type==='streamlit:setComponentValue').at(-1)")
    assert msg and msg['dataType']=='json'
    segments=msg['value']['segments']
    assert len(segments)==2,segments
    assert segments[0]['start']==4 and segments[0]['end']==9,segments
    assert segments[0]['muscle']==1 and segments[0]['force']==2,segments
    assert segments[1]['force']==3 and segments[1]['end']>17.4,segments
    screenshot=Path(__file__).resolve().parents[1]/'timeline_component'/'timeline_smoke_preview.png'
    page.screenshot(path=str(screenshot),full_page=True)
    # overlap cannot be saved
    page.locator('#start').fill('8');page.locator('#start').press('Tab')
    page.locator('#save').click()
    msgs=page.evaluate("() => window._testMessages.filter(x => x.type==='streamlit:setComponentValue').length")
    assert msgs==1,msgs
    browser.close()
print('PASS browser: endpoints, edit, two-click add, drag, save, overlap rejection')
