import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.exporters import CsvItemExporter
from scrapy.utils.project import get_project_settings

class MyItem(scrapy.Item):
    text = scrapy.Field()
    author = scrapy.Field()
    
class ScrapeArticle(scrapy.Spider):
    name = 'article'
    start_urls = ['http://quotes.toscrape.com/']
    
    def parse(self, response):
        item = MyItem()
        texts = response.xpath('//div[@class="quote"]/span[@class="text"]/text()').extract()
        authors = response.xpath('//small[@class="author"]/text()').extract()
        texts = [text.strip().split(',') for text in texts]
        authors = [author.strip().split(',') for author in authors]
        result = zip(texts, authors)
        for text, author in result:
            item['text'] = text
            item['author'] = author
            yield item

class MyItemCSVExporter(CsvItemExporter):
    def serialize_field(self, field, text, author):
        return super(MyItem, self).serialize_field(field, text, author)
        
# process = CrawlerProcess()

# To get the output in a CSV Format
settings = get_project_settings()
settings.overrides['FEED_URI'] = 'scrape_quotes.csv'
settings.overrides['FEED_FORMAT'] = 'csv'

process = CrawlerProcess(settings)
process.crawl(ScrapeArticle)
process.start()

