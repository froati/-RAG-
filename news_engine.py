import requests
from bs4 import BeautifulSoup
import urllib.parse

def get_samsung_news(limit=5):
    """
    구글 뉴스 RSS 피드를 사용하여 '삼성전자' 관련 최신 뉴스를 가져옵니다.
    네이버의 크롤링 차단 문제를 해결하기 위한 대체 뉴스원입니다.
    """
    query = urllib.parse.quote("삼성전자")
    # 구글 뉴스 RSS URL (한국어 설정)
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # XML 파싱을 위해 lxml이 있으면 사용하고, 없으면 기본 html.parser 사용
        try:
            soup = BeautifulSoup(response.content, 'xml')
        except:
            soup = BeautifulSoup(response.content, 'html.parser')
            
        items = soup.find_all('item')
        
        news_list = []
        for item in items[:limit]:
            title = item.title.text
            link = item.link.text
            
            # 언론사 정보 분리 (제목 끝에 ' - 언론사' 형식으로 붙는 경우가 많음)
            if " - " in title:
                title_parts = title.rsplit(" - ", 1)
                title = title_parts[0]
                press = title_parts[1]
            else:
                press = item.source.text if item.source else "구글 뉴스"
            
            # 구글 뉴스는 요약을 제공하지 않으므로 빈 텍스트 처리
            summary = "" 
            
            news_list.append({
                "title": title,
                "link": link,
                "press": press,
                "summary": summary
            })
            
        return news_list
        
    except Exception as e:
        print(f"뉴스 수집 중 오류 발생: {e}")
        return []

if __name__ == "__main__":
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

    news = get_samsung_news()
    if not news:
        print("뉴스를 가져오지 못했습니다.")
    for n in news:
        print(f"[{n['press']}] {n['title']}")
        print(f"링크: {n['link']}")
        print("-" * 30)
