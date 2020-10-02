import glob
import finacial companies
import companies

def find_file():
    return glob.glob("../input/**/*.", recursive=True)[0]

def run_sql(query):
    return financialcompanies.sqldf(query, globals())

print(find_file())

Company Info = financialcompanies.read_excel(find_file(), sheet_name="CompanyInfo")
print(CompanyInfo)
Employment Info = financialcompanies.read_excel(find_file(), sheet_name="EmploymentInfo")
print(EmplymentInfo)
Asset Table = financialcompanies.read_excel(find_file(), sheet_name="AssetTable")
print(AssetTable)
Stock Info = financialcompanies.read_excel(find_file(), sheet_name="StockInfo")
print(StockInfo)

CompanyInfo = run_sql("""
    select CeoName, Headquarters
    from CompanyInfo, Financial Companies
    where Companyinfo.CompanyInfo.ID 
""")

print(Company Info)

Employment Info = run_sql("""
    select NumberofEmployees, Median Salary
    from Employment Info
    where Employment Info = "E"
""")

print(Employment Info)

AssetTable = run_sql("""
    select *
    From Asset Table
    where Asset Table=Assets under mangement 
""")

print(Asset Table)

Stock Info = run_sql("""
    select *
    From Stock Price, Stock Name
    where StockID = "1"
""")

print(Stock Info)
