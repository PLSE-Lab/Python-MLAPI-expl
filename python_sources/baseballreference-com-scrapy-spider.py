import scrapy
from baseball_reference_spider.items import BaseballReferenceItem

class BaseballReference(scrapy.Spider):
    name = 'baseball_reference'
    allowed_domains = ['baseball-reference.com']
    start_urls = ['https://www.baseball-reference.com/leagues/MLB/2016-schedule.shtml/']

    def parse(self, response):
        root_domain = 'https://www.baseball-reference.com'

        for sel in response.xpath('//*[@class="game"]'):
            item = BaseballReferenceItem()
            item['boxscore_url'] = root_domain + str(sel.xpath("em/a/@href").extract()).strip("[,],'")
            request = scrapy.Request(item['boxscore_url'], callback = self.parseGameDetails)
            request.meta['BaseballReferenceItem'] = item
            yield request

    def parseGameDetails(self, response):
        item = response.meta['BaseballReferenceItem']
        item = self.getGameInfo(item, response)
        return item

    def getGameInfo(self, item, response):
        item['away_team'] = response.xpath('//*[@id="content"]/div[2]/div[1]/div[1]/strong/a/text()').extract()
        item['away_team_runs'] = str(str(response.xpath('//*[@id="content"]/div[3]/table/tbody/tr[1]').extract()).split('<td class="center">')[-3]). split('</td>\\n')[0]
        item['away_team_hits'] = str(str(response.xpath('//*[@id="content"]/div[3]/table/tbody/tr[1]').extract()).split('<td class="center">')[-2]). split('</td>\\n')[0]
        item['away_team_errors'] = str(str(response.xpath('//*[@id="content"]/div[3]/table/tbody/tr[1]').extract()).split('<td class="center">')[-1]). split('</td>\\n')[0]
        item['home_team'] = response.xpath('//*[@id="content"]/div[2]/div[2]/div[1]/strong/a/text()').extract()
        item['home_team_runs'] = str(str(response.xpath('//*[@id="content"]/div[3]/table/tbody/tr[2]').extract()).split('<td class="center">')[-3]). split('</td>\\n')[0]
        item['home_team_hits'] = str(str(response.xpath('//*[@id="content"]/div[3]/table/tbody/tr[2]').extract()).split('<td class="center">')[-2]). split('</td>\\n')[0]
        item['home_team_errors'] = str(str(response.xpath('//*[@id="content"]/div[3]/table/tbody/tr[2]').extract()).split('<td class="center">')[-1]). split('</td>\\n')[0]
        item['date'] = response.xpath('//*[@id="content"]/div[2]/div[3]/div[1]/text()').extract()
        item['start_time'] = response.xpath('//*[@id="content"]/div[2]/div[3]/div[2]/text()').extract()
        item['attendance'] = str(response.xpath('//*[@id="content"]/div[2]/div[3]/div[3]/text()').extract()).split(': ')[1]
        item['venue'] = response.xpath('//*[@id="content"]/div[2]/div[3]/div[4]/text()').extract()
        item['game_duration'] = response.xpath('//*[@id="content"]/div[2]/div[3]/div[5]/text()').extract()
        item['game_type'] = response.xpath('//*[@id="content"]/div[2]/div[3]/div[6]/text()').extract()
        item['other_info_string'] = response.xpath('//*/comment()[contains(., "Start Time Weather")]').extract()
        return item
