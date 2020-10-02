import scrapy
from charity_navigator_spider.items import CharityNavigatorItem

class CharityNavigator(scrapy.Spider):
    name = 'charity_navigator'
    allowed_domains = ['charitynavigator.org']
    start_urls = [
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=1#ltr-1',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=2#ltr-2',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=3#ltr-3',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=4#ltr-4',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=5#ltr-5',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=9#ltr-9',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=A#ltr-A',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=B#ltr-B',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=C#ltr-C',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=D#ltr-D',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=E#ltr-E',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=F#ltr-F',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=G#ltr-G',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=H#ltr-H',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=I#ltr-I',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=J#ltr-J',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=K#ltr-K',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=L#ltr-L',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=M#ltr-M',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=N#ltr-N',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=O#ltr-O',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=P#ltr-P',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=Q#ltr-Q',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=R#ltr-R',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=S#ltr-S',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=T#ltr-T',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=U#ltr-U',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=V#ltr-V',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=W#ltr-W',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=X#ltr-X',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=Y#ltr-Y',
        'https://www.charitynavigator.org/index.cfm?bay=search.alpha&ltr=Z#ltr-Z'
    ]

    def parse(self, response):
        charities = str(response.xpath('//*[@id="maincontent2"]/div/a/@href').extract()).split(', ')
        while len(charities)!=0:
            item = CharityNavigatorItem()
            item['charity_url'] = 'https' + charities[0].strip("[,],'").split('http')[1]
            request = scrapy.Request(item['charity_url'], callback = self.parseOrganizationDetails)
            request.meta['CharityNavigatorItem'] = item
            del(charities[0])
            yield request

    def parseOrganizationDetails(self, response):
        item = response.meta['CharityNavigatorItem']
        item = self.getOrganizationInfo(item, response)
        return item

    def getOrganizationInfo(self, item, response):
        item['charity_name'] = str(response.xpath('//*[@id="maincontent2"]/div[1]/h1/text()').extract())[8:-6]
        item['city'] = str(response.xpath('//*[@id="leftnavcontent"]/div/p[1]/text()').extract()).split('\\xa')[0].split(',')[-2]
        item['state'] = str(response.xpath('//*[@id="leftnavcontent"]/div/p[1]/text()').extract()).split('\\xa')[0][-2:]
        item['organization_type'] = response.xpath('//*[@id="maincontent2"]/p/text()').extract()
        item['overall_score'] = response.xpath('//*[@id="overall"]/div[1]/table/tr/td/div/table/tr[2]/td[2]/text()').extract()
        item['financial_score'] = response.xpath('//*[@id="overall"]/div[1]/table/tr/td/div/table/tr[3]/td[2]/text()').extract()
        item['accountability_score'] = response.xpath('//*[@id="overall"]/div[1]/table/tr/td/div/table/tr[4]/td[2]/text()').extract()
        item['cn_advisory']  = response.xpath('//*[@id="maincontent2"]/span/text()').extract()
        item['total_contributions'] = response.xpath('//*[@id="summary"]/div[3]/div[4]/div/div/table/tr[9]/td[2]/text()').extract()
        item['other_revenue'] = response.xpath('//*[@id="summary"]/div[3]/div[4]/div/div/table/tr[12]/td[2]/text()').extract()
        item['program_expenses'] = response.xpath('//*[@id="summary"]/div[3]/div[4]/div/div/table/tr[16]/td[2]/text()').extract()
        item['administrative_expenses'] = response.xpath('//*[@id="summary"]/div[3]/div[4]/div/div/table/tr[17]/td[2]/text()').extract()
        item['fundraising_expenses'] = response.xpath('//*[@id="summary"]/div[3]/div[4]/div/div/table/tr[18]/td[2]/text()').extract()
        item['payments_to_affiliates'] = response.xpath('//*[@id="summary"]/div[3]/div[4]/div/div/table/tr[21]/td[2]/text()').extract()
        item['excess_or_deficit_for_year'] = response.xpath('//*[@id="summary"]/div[3]/div[4]/div/div/table/tr[22]/td[2]/text()').extract()
        item['net_assets'] = response.xpath('//*[@id="summary"]/div[3]/div[4]/div/div/table/tr[24]/td[2]/text()').extract()
        item['compensation_leader_compensation'] = str(response.xpath('//*[@id="summary"]/div[3]/div[11]/div/table/tr[2]/td[1]/span/text()').extract()).split('\\r\\n                    ')[1][:-23]
        item['compensation_leader_expense_percent'] = str(response.xpath('//*[@id="summary"]/div[3]/div[11]/div/table/tr[2]/td[2]/span/text()').extract()).split('\\r\\n                    ')[1][:-23]
        item['compensation_leader_title'] = response.xpath('//*[@id="summary"]/div[3]/div[11]/div/table/tr[2]/td[4]/text()').extract()
        return item
