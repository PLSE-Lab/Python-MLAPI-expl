import glob
import pandas
import pandasql

def find_file():
 return glob.glob("../input/**/*.xlsx", recursive=True)[0]
def run_sql(query):
 return pandasql.sqldf(query, globals())

print("------- BRAND DATA -------")
brand = pandas.read_excel(find_file(), sheet_name="Brand")
print(brand)

print("\n------- FOUNDER DATA -------")
founder = pandas.read_excel(find_file(), sheet_name="Founder")
print(founder)

print("\n------- PRODUCTS DATA -------")
products = pandas.read_excel(find_file(), sheet_name="Products")
print(products)

print("\n------- BRANDPRODUCTS DATA -------")
brandProducts = pandas.read_excel(find_file(), sheet_name="BrandProducts")
print(brandProducts)

print("\n------- INSTAGRAM DATA -------")
insta = pandas.read_excel(find_file(), sheet_name="Instagram")
print(insta)

sql1 = run_sql("""
 select *
 from products
 where Price > 40
""")

sql2 = run_sql("""
 select brand.BrandName, products.BestSellerName
 from products, brand, brandProducts
 where brand.BrandName = 'Kylie Cosmetics' 
 and brand.BrandID = brandProducts.BrandID 
 and products.ProductID = brandProducts.ProductID
""")

sql3 = run_sql("""
 select FounderFirstName, FounderLastName, FounderNetworth
 from founder
 where FounderNetworth like '%B%'
""")

print("\nAll products costing more than $40")
print("\n",sql1)

print("\nBest selling products made by Kylie Cosmetics")
print("\n",sql2)

print("\nAll founders who are billionaires")
print("\n",sql3)



