import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Exercício 1
products = pd.read_csv('../input/olist_products_dataset.csv')
products['qty'] = 1
qty_products = products.groupby('product_category_name', as_index=False)['qty'].count()
qty_products = qty_products[qty_products['qty'] > 500]
qty_products.to_csv('exercicio1.csv', index=False)


# Exercício 2
orders = pd.read_csv('../input/olist_orders_dataset.csv')
order_items = pd.read_csv('../input/olist_order_items_dataset.csv')
orders = orders.merge(order_items, on="order_id")
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
orders["delivery_time"] = (orders["order_delivered_customer_date"].dt.date - orders["order_purchase_timestamp"].dt.date).dt.days
orders['purchase_year'] = orders['order_purchase_timestamp'].dt.year
orders['purchase_month'] = orders['order_purchase_timestamp'].dt.month
orders = orders[(orders["purchase_year"] == 2017) | (orders["purchase_year"] == 2018)]
orders_kpis = orders.groupby(['purchase_year', 'purchase_month'], as_index=False).agg({"price": ["mean", "sum"], 'freight_value': 'mean', 'delivery_time': 'mean'})
orders_kpis.to_csv('exercicio2.csv', index=False)
