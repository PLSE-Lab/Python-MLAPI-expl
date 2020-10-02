#!/usr/bin/env python
# coding: utf-8

# # Diabetic Retinopathy 
# 
# Diabetic retinopathy (DR), also known as diabetic eye disease, is a medical condition in which damage occurs to the retina due to diabetes mellitus. It is a leading cause of blindness. Diabetic retinopathy affects up to 80 percent of those who have had diabetes for 20 years or more. Diabetic retinopathy often has no early warning signs. **Retinal (fundus) photography with manual interpretation is a widely accepted screening tool for diabetic retinopathy**, with performance that can exceed that of in-person dilated eye examinations. 
# 
# The below figure shows an example of a healthy patient and a patient with diabetic retinopathy as viewed by fundus photography ([source](https://www.biorxiv.org/content/biorxiv/early/2018/06/19/225508.full.pdf)):
# 
# ![image.png](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxITEhUUExIVFRUXGBcXFhgYFxcYGRoaGRoXFhcYGRUYHiggGRolGxUWITElJSkrLi4uFx8zODMtNygtLisBCgoKDg0OGxAQGy8lICYtLTU1Ly0tLS0vMDUtLS0tLS0vLS0tLS0vLS8tLS0tLy8tLS01NS0tLS0tLS0tNS0tLf/AABEIAKUBMgMBIgACEQEDEQH/xAAcAAABBQEBAQAAAAAAAAAAAAAAAgMEBQYBBwj/xABFEAACAQIDBQUECAMFBwUAAAABAgMAEQQSIQUGMUFREyJhcYEykaGxByNCUnLB0fAUYoIzkrLh8RVDU1RzosIWNGODs//EABoBAQACAwEAAAAAAAAAAAAAAAACAwEEBQb/xAAwEQACAQMDAgMHBAMBAAAAAAAAAQIDBBESITEFQVGx8BMiYXGBkdEyocHhIzNCFP/aAAwDAQACEQMRAD8A9xooooAooooDM764nFp/C/wtyxn762urosUrmNjY5c2UKG5MVqh2PvRjcsAaJiWjzMHil7U5klkLl1sihCioVIuSeIuoPodNYnEogzOyqOpNqN4MpZ4PPBvVtURs5ghLKrPlEE/ey4eHFZQc/E9q0V7e0pNtMtStq7yYwzTQxoyqrw5XWJwwtiIEkUklg6skjG9l0BPjV9id78MpspaQ/wAq6e9rCoEm+33YD/UwHyBrXldUY8yNqFhcS3UH9dvMg7QxWOG0WKtKMOMRBHfigVogzL2PZ6qznL2gfusdRYGqxt7NpSRhzH2Qy4nMiwyZi4wxkjizXJV0cEZvtGwsCLG4bfx1YBsOMp0BEnPoQVqwh30i+3HIvlZvlrWFeUX/ANE5dOuYrOnyKqfenGpmvEMp7VUbsZWKmOaKJWk7wDB1lLX7tshOovbWbt4558LDNInZySRqzpYjKxGq2bUWPWjA7bgl0SVSfunun3GrGr4yUllPJpyhKDxJYZ2iiipEQooooAooooAooooArK764pkfDDtp4YmM3aNCpZriMmO9kb7XAW1Nh4VqqKA8zw29e0IzGssDmRnwwkDROVs0WD7bKykCMhppGIs3sNwsbdj3rx8auvZ9rZ3HaNDKOyHbTKgkBZRLdFSxUiwIJvpf0quM4AuTYDmaA87k3kx6yylo2OVS6RCOSyg4aJ7FgLygSM/RrowHQWO2Mfi5cJhmhYtJJiAjmNZMOHQdqCfrEdokORTcgjUWNiDV1i96cLHp2mc9EBb48PjVbLvsn2YWPmwHyvVErmlHmRtQsq891B+XmUmy95No/VQmMljC5Z3jbOJPr9LiysYjEiHu2fMGBW4BTBvNtARllQSN2AlLPDLlZkwySsqotirM5Zeet9OVWWI36kUXGHBHP6w3t19mpWH34QgFoXAP3SG/SoK8ov8A68yx9NuUs6f3X5KrFb54xO1ZoUUKyqA0ct4r4qHDJ2jXtKZI5GlXJa2S2t71xN7sczpGIdXSYFxBKBdVxBhnQs3sP2MfcIuO1AzcK00e3sHiAEdl4q2WVbd5WDqe9pcMoI6ECrxWB51sQnGazF5NSdOdN4mmvmea7N3m2giPK9pkvmOaGRCFjw0E0mQkgAH65QCPb5/Zrd7AxUk2HjlkAVpBnCi4srEmMG/2shW/jepWLwscqFJEV0bRlYAqeeoPGngKkQO0UUUAUUUUAUUUUAUUUUAUUUUAU3NKFBZiABqSTYDzNN43FLGhdzZVFyf3xNeX7wbblxbkXKwDgnC55FiOJ+A+Na1xcxorfnwN2zsp3MttkuX67l7tjfcsSmGAsNDK3M9EX8z7qzU0zyNmdi7dWN/d09KZWK2lrCnFFcKtczqv3ment7SlQXuL69xQ01pCknjoPnQq3PhypwVTqNjgR2CnxHmaZYsnVl5g8QPA8xUi3SuG/wC/0pqMpnTYjSrHZ28OJgIs3aIOKOT8H4qfeKpmw33WK9NdPceVLikPsvo3wI6j9KshVlB5iyupRhUjiSyviem7B3hhxQ7hyuPajawdfTmPEaVcCvGmVlIeNisi6qw0I/UVv90951xKhHsswFyOAcDiVHXqK7Nreqo9MufM85f9NdFe0p7x/df0aaiuXrtb5yQooooAooooArhNBNefb370O5MOGfKODuOJHMKeQ8eflVNavGlHMjZtrWdxPTD7+Bcbwb4pE3ZwgSy89e4n4iOJ8B8KxmP2nNMbyOza8Bog/p4fOoccAUd0Wrt7X/WuFXup1eePA9Ra2NKgtll+L5/oczcLWvcVztDmsD5n/L0pOUtyt1/Q1x1YW/L9ePrWvk3MIWFF7fmdeFcMRFyhI6qeHp0pMLkknnp5jTh8Kcs1tD+tYyMioZQwvax4EHkRUrBbSnh1ikI/lPeQ/wBH6WNGEkgVHzDvtz4VAVimjG68m5+TfrVqk44lF7lTjGpmLjt8e56BsDe+OZhHKBFKeAJ7j/gY8/A6+daYGvHJoQwsa026e9RQiDEve1lSQ+PAOenj766ltf6npqfc4d90rSnUo8d1+DfUVwGu10zhhRRRQBRRRQBRRRQBXCa7WZ362t2MORTZ5e6PBftH8vWoVKipxcn2LKNJ1ZqEe5lN79ufxEmRT9SjWH854Fj4ch69apl0PgfnTRcAeooYE25C/rb8q81VqOpJykeyoUo0oKEeCQ01tPd1pLO1uHxHypK6X+P+tKUc6oZsIcjm5HQ9DToamreF66E6G3nqKGNhw0CkC9tb+mtdAY8PjagBpADXHjVx8iOI8qcWO3j86bcDy8uNEE/ARFLY5WPe5H7w/WoJJR45FbKVY5SOTA6VJnXMLEnwJHyYVFxcfcK34d5W8RrrVsHuYnhrHievbtbZXExBtA47rr0P6HjVvXkm6G1TDOjm4jkCq48+Deh+BNetLXobWt7WG/KPH3tv7GphcPj8HaKKK2TTCiioW2cesELyt9kaDqToo9SRWG0llmYxcmkjMb9bdKj+HjNiR9Yw+yDwXzPy86w6aAEctT+dSYF7YuzyAMxLG/MnU00MM5UkWAA49fKvO3FSVWWp8dj11pShQhoXPf4nTMBz0NJMjch79P8AOmoxbre1tfy6CljXyrUZvoewaZiQzhONr8OvEHjQ4AJAIPXn6611Jbcr3o7IcRofC1Zclgj33IsuEs4mF8wUhgDoy8SCOZHEevWpq2axGt+FufSkqG56+X6UzC7RuAtwCTl0Fg3ErryOpHqOlZzq5IY0vK7kmU246eFN51Olrg8f8xUqVmaxa2YUxKBzFv3yqL2exKL23J8WFiEeQXJGubp4Xqh2pDfODyAJ8RzqX/ENlsrG3W1/86gxtqbnMDob6n/TWrtSeNiEYyjnLzk9B3C292iiB2u6i6MeLILaE8yt/dWyrwrZWKeJlZD34n7p6/5WJB9a9r2bjFmiSReDqCPDqPQ6V2rKvrjpfK8jzPU7ZUqmqPD8yVRRRW8c0KKKKAKKKKA4a8k322l2mLcC5Edox001b/uJ91esTyZVLHgAT7hevBllLkueLEsfU3PzrndRniCidbpNPM5Tfb+SVGL2vqb+7yp8jUeR/f76UzEKeYmuHI9GmdKUvMBpeo5auFredRSMvJKEgpyNhVcJPGpET1lowmTlFKNNI9KBqOSQvJ1oEfhRnqVglBOtZISlpWREeH0uRVftaIFSemvxGlaAgE25VU7bh4hTcAi46g+NTSK6VTMsMhxyi1rXH70r1PdrGGXDRsfatla/VdPjYH1ryuO4+ySeOugrcfR9iWImRgBlZWFjf2gR/wCNdDp82quPE5/V6SdLUuz/AKNhRRRXcPNhWB+k3aVuygF9frG9O6vxzH0rfV45vxie0x8vRMkY9FBP/czVp30sUseJ0Om09VfL7blYhuNeHT9amO5yBbmwtUKIeNSS1hxtyvXAkepXYWqXvx6+6nImC63193xpqxIPnw9CfdpSQ1r3sfOq+STeR0tc24+puSfKlo+oBHEfpxqEZCG1uDyA0/YrhxRWVoipy2DI/Xk6W8NDf+bwrOM8EHLDSLTXmD+/CuyxKy2N/TQ9QR4ggH0rkE1jc/DjQZCT4fvWoptMzzsJhkJBDWzLoRwv0YeB+Go5V2RDY90f3hTeIXUMASwHD7y8189LjxHiae9sKVIIup1GhHG9T2e5HLWw3FDcqOAIJ58efCkDDakE68j15MflVoAL2sGBs3Ai3RgDxHlXdoRgAAHUjjpobHQdKnjOyK41cyKDB6AW4i4Pv1NehbgY0skkZFgpDL5Nx+I+NeeICLAob6c9Lak69b1qNy8U64pFIAV1ZfaJPDMOPlW1Zz01l8SjqdNToy+G/wBj0miuCu16A8mFFFFAFFFFAV+8D2w05HKKT/Ca8Pwz8K9z23Fmw8y9Y5B71NeC4fgPKuX1FbxO50j9Mvmi1QinA96gpyqReuRJHcQ63GmJDenQ3Go+JQMcg8C34eFv6tR5XrEUJPBGwUjNdiuUFjkHMroAxPibkeFqsIr0gRG9OoLVlvJGLeB8NUhG0piJbU7a2o4dKqZZEcvXYnsa5SawSayiXisUVUsp15eZ0FU22oXOHl75zlGu/EjQ3sKeDFnFjoup6X5Vza09opPwN8QQKuhlSRRJYi0SI5VZbk3sLedqvvo2mJxE4J+wh18GP61h53KvppmF/UaGtv8ARcCZJ2I4JGAfMsfyrds1/mj67Gj1H/TL13PRaKKK7x5gK8N29LfF4gn/AI0nwYj8q9yrwzeWPLjcSP8A5WP97vD51odQXuL5nV6T/sl8v5OYfUgHhcVdbTwKRxAjVidOnX8qzkY08eXSpZnYqASbDhfhXG2SeUd9puSaeB0P3DwDHj4rzH76UxIelOxvccLgc/Dnf4VDjTKzZnJRzdbj2DzUniVY8OnDmKrjHktlLT2Bgb3v/pXZQbZhxU5vP7wHmt/W1L7M08iWotiDeR2Nja4N9Lj1qVCw14nyqBhBYstuFio/lb9DcegqcCfsnUfvWoTXZEovKFtwvb/WoN2VrFbKxup07rnUjwvqR436ipqsSNeV6bmiDKVPA8fLw6Hx5WpF45JOOV8RazcWW4tlUak8eP6+lRJMb9YgkJvLn7P7oKAadcxGvoaQXJ+r4NfW3Mff8Li/qCK5jswOYfZykW/l1+IzD1q6K3w/XgVPaOV68SVh5QWOb1vxU+XTXjUvYs1sdh7cO0t7waocZPe0g4X49VOg/I1a7m3bGQC17PfysrGrqK9+PzRRdv8Axy+T8j2QV2uCu16M8iFFFFAFFFFAJcXFeB4zDGKWSI/Ydl9x0+Fq99NeU/SZswx4kTD2Zl1/GgsfeuU+hrRvoZgpeB1OlVdNVxfdeRmYwalR2NRIn0p+9u9r0PlXEkj0S3HZ5Aouwvy08dALdb6UQwiw1GZu8x8fDwAsBUcuWa4tlU2Hi3An04ed6eYWN7kdeFvOotYWCSWdyUFFDGwNhf8Ad7DxplFPU0qJjwIvUTOk7s2ZnijZlszKGZehI1FTD5VAwbkRpofZHTp51J7Runx/SkuWZgvdWRwMbcKTMWsLaEn9mmJcUV0IuTwA/elP4drDMzAt8AKwkWDqRhQAPWq3ahzJJ0CsT520qZJIW0XhzP6VF2gAInAH2W+RqUX7yIyjiLGNpRnRhxX5c69D+jHDWw7ycpH7v4VAHzLVg5QWIAFySAo6k6AV7BsHZ4w+HihH2FAJ6n7R9Teun06DlNyfb+TkdYqKNJQ7t+RYUUUV2TzQGvIPpGwpTHM3KVEf1HcP+Ee+vX6xP0obMz4dZgNYm1/A2h9xyn31rXcNVJ/A3en1NFdZ77evqebxj1qRGeRqLC1SOPW/EVwJI9SsEmMkDTlw/Slo0ZBuAQQQRyueAI6XqOJLi4+R+VdeIgA3GouCOB8P8jVTRPCYQDIcp1+6eoHI/wAwHvGvI1JU2qKyZhqT100II4eRFEEjAlWsSLcNLjky+B6ciLdCZNZ3MaexIlAXK9yMp72n2T7VvKwP9NSwdDpqT16VFZ/A+tj6caRhZGtltqump5cVPu08wai1lGVHDJakjlpRc20FqYkmZRc2tXcPIXN2ICjgvP1rCRaAhIAfjJ8xzXw8PECkYhg1ipve3uHy41cWR4mK2zCqML2bdQ51PRzy8A3z/FVjWEimLy2yOIB2Zj+7oPLivwIHoa0n0YQF8Sz2/s0Ib8RNl+Gas9O9nB5MMp8xdl+Gb4V6T9HWy+ygaQ6NM2f+kDKn5n1rcsoudVP6+vqaXUpqnbyXd7evoawUUUV3jygUUUUAUUUUAVTb17HGKw7x/a9qM9HXh6HUHzq5oNYlFSWGShNwkpLlHz8LqxVgQwJUg8QRoQfWlSSnRV0Y8+g5t6fMit39I27PHFxDgCZ16qP94PEAa+Avyrz/AAhv3jxPAdByHnzPifCuDWoulLc9TbXEa0E4/X4E1O6LAadOlKje+h5UkGuGx5XrVaN1eA8j208dKdR9fdUPIOGvvpcYYcx63qLRPA/gW+rXyFcMxYlUPm3IeXjUDBSM6AXso004nU1OimCjKBcjkPz6VmSw2YhwhSYcXsLn7xPE+tKkjUA6VxTbnqeJomewB8dag+S5McMltKibRf6t/wALfKlO16VDgJJxIijRUZnbkqgHW/U8hU6cHKSSKq1SMINvY1O4WyDJN27DuRkhfF+v9IPvPhXo4qPs7CJFGqILKo0HzJ8TUmvR29FUoKJ468uXcVXPt2+QUUUVeaoU1iYFdWRhdWBUjqDoadooDwnbezWws7wtewPcP3kPsn3aHxBqfj9nBIkkU+1a4Neh767tjFxd2wmS5jbr1QnobDyOteQQyzd5Jrhkd1yk6qAbAEcjXFuKCpt7bdj0tpde2Ud91yTATy9fLjXVlsbEaHr+tWcCwNhzmv2o9kCqyNGYHu3txrSlDB0Yz1J9sCyQCenh86XIwIGgzAkq1z7iOangR+YFRgnmPWlIhGot43v+VV4LHHKwyVBMGvbQ/aHRuYvzHQ9DTbOBKnAZu6RxsD7JPgGt/eNQppJM3ctmGhtzHTz6Hr5mn4WSxvchhY/ePIi3Ig+4ipKKW5jDe2dy7xGwwiBzJmkJsOg8hUEwKOVMYfFOy99j3TYjx+95kEH1pXbXzAHW1YqYb2WCVFTx7zyLjksvhSZ+8pBFwRY+tNmS49KsdlY0KBGsRklZrKBre/y0/WkIZfJmrPTFtLJC2Vs2TEv2I1ZBnY9VU3B8CxGXzJr2fBMpRSnslQV8rC3wqi2ZsE4ePOtjPcu1tA1+MWv2bWt4gHrVhsKdSrIvBTdfwPdlFuVjmW3LJXoLSh7KG/LPJdQu/wD0VNuFx+S0oooraNAKKKKAKKKKAKKzu098sNBM0D586mAWCg37Ziq5ddbWu3QVKXeXCtos6X74sc2hQXbMLXAsQdbXvpQD2JPayCP7CWaXxOhSP/yPgFH2qx+925Ju02FW5Ny8XDXm0fifu+7pV8m8OEw4CPMD3ZJHksSt17IuWI4aTxkDgF8BWiVgRcG4OoNV1aUakcSLqFedGWqB8/5jcjgRoQRYg8wQeBpwSWr1/eDdXD4rvMuSTlIlg3ryYedeebb3JxsP9mgxC9UsGHiYyfkTXJq2c48bo9Db9RpVNpPD+P5KGSYDiQKYbGFhYcOF7anwApM0YiNpFIfo6kH3MKVBbieJrVccHQUsisECFAvYa+ftHif0qYrAW0qLhSMvHm3+I080ijiR4VGS3EJPCHsxodtKlbP2Ripz9VA5H3iMi/3m/K9bLY30fKLNiXzn/hron9TcW+Aq2la1KnCKK9/Sord7+C3Zkt39hzYsjs9EBs8hHdHgPvNbkPWvSpdlR4bBTRxjTs3JJ1ZjlOrHrVzBCqKFVQqjQACwA8AKi7f/APbT/wDSf/Ca7Fvaxpb8vxPO3l9O4eOI+H5JqcB5UqkpwHlSq2TRCiiigCiiigOGsXtndRcUryJZJxJKAx4MAxsr2+B4jxra1B2T7L/9WX/G1QnCM1pkWU6kqctUXueJ4yGSFzHKpRxxU9OoI0ZfEaUiLEMoNiRfQ+Ne27X2NBiUyTRhhyPBlPVWGqnyrzzbu4E8QZ8O4lUXOVyEcAa+17Letq5daylHeG6O/bdTpzWKmz/b18zJtLzvUrZW8CPHLFbIFFlkIvnOt8o5jhrzvoNKqcRgnXvYiN0U6gMpAbpqdMvz8uPYnDG+lhwHLztWsl7POVubzxUSw9iZsmcxuHtfW+vXwHIcal7ZdRKZVsQ1i4HuuOpAGvUeIqEtutKzAcxaq9WFgscU5as7jYezAg6Pp1F/snw0LD1WpSNYn41HhwcjjLFG7IxABCmyMSMvePdCliCLnQ+B02O7248kyh8RIIxchkTV8ymzKW4LqDwvU4286n6UV1L2lRypvBhX2oolaIggjUnllCGR29LDT+YV6HurtHAYVQzO7SMcjyGNgqNqxiv9khUZyde6ua9rVrsPuxg0XIMNERct3lDEkoYyxZtSShKk9DanBu7hP+Wi/szFqgPca4ZDfiCGYf1Hqa61C0hTxLueeu+oVK2Yr9Prkr8RvphFtcyHNF2q5Yycy8rAai471zYW1vTGA21hu0eeNpOzbs0b6s5DJK8YUBx9q8q3HDvsetriXd/CM2ZsPEWyhL5RfKBYAHlYaCkru1gwCBhoQGUIbIouotZf+1f7o6Cts55X4ffTDtL2feGaQRo1tGJUHUcVGY5fS/CkYXffDuFcLKUdkWMiNiTnjEqsUtcKUOa+tlBJtVnHu3g1ZXXDQhltlIRQRa1tfCwt0pM27GCcKGwsJCjKAY1sFCrHlt93IirbooFAVZ36w5yFVfI18zsLKmWQRMGtc3ubi2h61JO+OGDKjdqrFlUgxtdMzQqpf7oJxENvx+BtOXdzCDLbDRdy5XuDQsQWI8yoPpSotgYVQAMPHYajujSzI49zRxn+hegoCyorlqKAp8fuvhZpe1kjzSad7Mw4BRpY6aItMYTc7CR+wri9w31snfBAWz97viwGh8TzNaCigM3NuPgnUKyObZtTLJchxGrAnNqCsMY14ZdNa0MMQVQo4AADyAsKXRQBXK7WT3oxuNTEIIM3ZBUaSyBhrKEYkZCzgJclVZSOOtAaXEYRJBZ0Vx0ZQ3zqql3QwLccLF6Ll+VqzcG9W0JCt8KYlIk0COz5lfC2W5UqCFlmB4huyYi1rB8b240LEWwRDO8VwEmICSJEzXJXR0MjX0P9mdBraLinyicako/pbRP2TujgSrXwyG0ko1zHQOwHE1dYTYuGj/s4I18Qi399qzMu3Magwf1ZcywkyjsnB7UvCosVUiNgryNZrA5T5iLhN7scI1D4U5hHDncxTjIWEGaV0VNVYySWRLleyOa3eyFCK4Rl1ZvZyf3N+BUHbm0Rh8PLMVzdmpa17A26tyHU8hc1n9k7fxk2JhR8O0CFM0ilHJuYo3BMhUKozs6WvmvGbgcBrWUEWIuDxFSKzKjfERl0mjBdHKsYGV0sEhkZ7uVOgnW6gE6aA3FMYbfCPE5YHhdTPCrlQyXRHJjZmdmAOpSwW7d7hoa0P+wsNeMiCMdlm7MBAFUsVYkIBYNdFN7XFqe/2XB3fqYu62Zfq17ranMNNGuza/zHrQFHtTfWDDyPG0cpKMIwVCENJliYILtcG0yG5AHHWupvvAXVezmGcxLGxVQGklWJ1iAzZg4SUMbgABXN9Kvp8BE4IeJGDXzBkU3uADe410UD0FKGEj07i6EEd0aEDKCPELpfppQFdu9vDHjA7RK4CEC7i2YEXVlsToehsRzAq4pnD4VI82RFTMxZsqhbseLG3EnrT1AFFFFAFQdkew3/AFJf/wBGqdVZgJlSFmYhVEkpJ/8Asf4+FAWEsqqCzEAAXJJsAOpNVyxmchnBWLiqEWL9GkHJeYX1PQdihaUh5QVUEGOM9eTyfzdF4Lx48LICgEvECLEAjoRcfGqrEbr4JzdsLET+AA/C1XFcNYcU+SUZyj+l4M6N0Nnf8vHrw7zeWmvWpuG3dwiG6YaIH8APzrDbP+jnFwgdnio1ZUEcbAMezGeOZrAjX63tj5MvSrufdvGNFHaUqyJl7MYvE5WvJmZWxFu0N00z2uOQqKpwXZE3WqPZyf3ZrmgUqVKjKQQVtoQdCLdKr9mYWSKR19qJgGVr3NxZcrA6k5QuvPLrre+Xm2LjYi0mI2giRZcOjsZZI75JcOXJubRsyJOl1IzdoL2peB3d2ijF/wCM7UWhyAyyZWCFLq3dNtFfvi5btO8NL1MqN1ei9YfBbv46OSDPiJJMzr/EMJGKCKOOMgBWIs7TIdQDdZHueFNYPdrGt3Wx1xHKcwWeYvZpMJIyyOLd4wpiABYAduv4qA3t6L1jt39k46GQSPOuIRo4UN5XOoEatIlwFAADt9osWGo1pnau7GOlknIxV45G0jMsiKUIYAHIt48lwe6bPYhqA296L1g8TuljyzhcYUQrZAssigWgaNBlUdwJLkbunvC9xe1TYd38as+HYYm8UUkrMDLMWZXaQhGDXDizLa+oK6G1gANhRRRQBRRRQBRRRQBRRRQBXK7RQFPt3EYpGi/h41cO+SS/+7BsRKdRdVCuCOJLLwsaz2C2xtYtCZMMoDSlZFCNdVsl+8SBlUl+99oLprodzRQGAj2ttd1CnDhbiUFwjLche7lBJKWubFhqRbUam83axmMaWWPERkIqx9m+Ui5IswJJ7zaXJAtrx5DR0UBy1doooAooooAooooAooooCFtbaKwR52DNqqqqi7M7sERVGguWI1JAHEkCoE+9eGjQtMWiYRtI0Tqe0AXOSCq3BJEbkAE5gpIuBerLaWAjnQpILrdToSpDKQysrLqrBgCCOBFVUm6OEb2kc90qbyyHNcSLna7avaaQZjr3vAWAdw+9OEdiolAYMVswZTddeY4cgeZBA1FQMFtDChO3kxMZjWSUx6lUU2eck5uL9mS1/u8ONzUDFbJu0paY99xIT2+W8VizuvAIhfPmOgLE05iMfsox9m0cjxFhiGOSUi0EcZjkt7RRkiUAgENYg8TQGm/9R4TNl/iI81mPtadzPmu3AW7OQ2vwRjyNL2ftyGZ8kTZu5nuAQLZihBvqGBHAiqnZm7Wz5YYnjjYxdnkRWaTKVyyR3ZCdXyyyLmOve48Kttm7DhgYugbOQQzM7OzXOa7Fjqb8+gAoCzooooAooooCj3v2G2Mg7ESdn3lY3BIOW9gcrKws2VhZhqgBuCQYGI3Wlf8AhC2KJOHWzdzIrkFCGCIwCGyZSOGV2HA2rV0UBh03JnWNYxjWFi2Z8r5yGkhlGU9poV7IoCb2QgcjdqHcKRRGBiVXLOs75YmF8q4dCATISMywOG1N+2bpY72igPPcF9HciQvEcVcNHHGtlkUAIYzYgSar9WbDgO0fiCQdxszDtHFHGzZ2RFUta2YqAC1uV7XtUqigCiiigCiiigCiiigCiiigCiiigCiiigCiiigCiiigCiiigCiiigCiiigCiiigCiiigKKXdLBs4cxDQsxFzlYtl9oHiBlsBw1OmtPru3hALCBRqx0uPaXKQCDouU2A4DlaiigJ2AwMcKCOJAiC9gOpJZj4kkkknUkk1IoooAooooAooooAooooAooooAooooAooooAooooAooooD//2Q==)
# 
# An automated tool for grading severity of diabetic retinopathy would be very useful for accerelating detection and treatment. Recently, there have been a number of attempts to utilize deep learning to diagnose DR and automatically grade diabetic retinopathy. This includes this [competition](https://kaggle.com/c/diabetic-retinopathy-detection) and [work by Google](https://ai.googleblog.com/2016/11/deep-learning-for-detection-of-diabetic.html). Even one deep-learning based system is [FDA approved](https://www.fda.gov/NewsEvents/Newsroom/PressAnnouncements/ucm604357.htm). 
# 
# Clearly, this dataset and deep learning problem is quite well-characterized. 
# 
# # A look at the data:
# 
# Data description from the competition:
# 
# >You are provided with a large set of high-resolution retina images taken under a variety of imaging conditions. A left and right field is provided for every subject. >Images are labeled with a subject id as well as either left or right (e.g. 1_left.jpeg is the left eye of patient id 1).
# >
# >A clinician has rated the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:
# >
# >0 - No DR
# >
# >1 - Mild
# >
# >2 - Moderate
# >
# >3 - Severe
# >
# >4 - Proliferative DR
# >
# >Your task is to create an automated analysis system capable of assigning a score based on this scale.
# 
# ...
# 
# > Like any real-world data set, you will encounter noise in both the images and labels. Images may contain artifacts, be out of focus, underexposed, or overexposed. A major aim of this competition is to develop robust algorithms that can function in the presence of noise and variation.
# 
# For a simpler project, I will perform binary classification of Referrable Diabetic Retinopathy (RDR), which are cases with stage 2 or above. This sort of classification has been done in previous research, including Google's research paper as mentioned above.
# 
# ## Preprocessing of the original dataset
# 
# The [original dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) is quite large (35 GB). Therefore, the data was significantly resized before being passed into the network. The data was downloaded onto a computer and cropped to get rid of as much black space as possible. This was done by applying a threshold to the images, find the largest contour, and finding the enclosing circle. This gives the circular mask. The radius of the circle is used as the width of the image, and the height is kept to maintain the aspect ratio, unless the it can be cropped without cropping out parts of the image:
# ```
# # import the necessary packages
# import numpy as np
# import cv2
# import glob
# import os
# from tqdm import tqdm
# import math
# from PIL import Image
# files = glob.glob('D:\\Experiments with Deep Learning\\DR Kaggle\\train\\train\\train\\*.jpeg')
# 
# new_sz = 1024
# 
# def crop_image(image):
#     output = image.copy()
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
#     contours,hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         print('no contours!')
#         flag = 0
#         return image, flag
#     cnt = max(contours, key=cv2.contourArea)
#     ((x, y), r) = cv2.minEnclosingCircle(cnt)
#     x = int(x); y = int(y); r = int(r)
#     flag = 1
#     #print(x,y,r)
#     if r > 100:
#         return output[0 + (y-r)*int(r<y):-1 + (y+r+1)*int(r<y),0 + (x-r)*int(r<x):-1 + (x+r+1)*int(r<x)], flag
#     else:
#         print('none!')
#         flag = 0
#         return image,flag
# 
# 
# for i in tqdm(range(len(files))):
#     img = cv2.imread(files[i])
#     img_cropped, flag = crop_image(img)
#     if not flag:
#         print(files[i])
#     height, width, _= img_cropped.shape
#     ratio = height/width
#     if width > new_sz:
#         new_image = cv2.resize(img_cropped,(new_sz,math.ceil(new_sz*ratio)))  
#     else:
#         new_image = img_cropped
#     cv2.imwrite('D:\\Experiments with Deep Learning\\DR Kaggle\\train\\train\\resized_train_cropped\\'+os.path.basename(files[i]),new_image)
#  ```       
# The folder `resized_train_cropped` contains the images, and `trainLabels_cropped.csv` contains the labels for the images.
# 

# In[ ]:


import os
files = os.listdir('../input/resized_train/resized_train')
print(len(files)) 


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


print('Make sure cuda is installed:', torch.cuda.is_available())
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)


# # Reading data
# Here I am going to open the dataset with pandas, check distribution of labels, and oversample to reduce imbalance.

# In[ ]:


base_image_dir = os.path.join('..', 'input')
df = pd.read_csv(os.path.join(base_image_dir, 'trainLabels_cropped.csv'))
df['path'] = df['image'].map(lambda x: os.path.join(base_image_dir,'resized_train_cropped/resized_train_cropped','{}.jpeg'.format(x)))
df = df.drop(columns=['image'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df['level'] = (df['level'] > 1).astype(int) # Disease or no disease
df.head(10)


# The dataset is highly imbalanced, with many samples with no disease:

# In[ ]:


df['level'].hist(figsize = (10, 5))


# In[ ]:


from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df,test_size=0.2) # Here we will perform an 80%/20% split of the dataset, with stratification to keep similar distribution in validation set


# In[ ]:


train_df['level'].hist(figsize = (10, 5))
len(val_df)


# In[ ]:


df = pd.concat([train_df,val_df]) #beginning of this dataframe is the training set, end is the validation set
len(df)


# In[ ]:


bs =16 #smaller batch size is better for training, but may take longer
sz=512


# Here, I load the dataset into the `ImageItemList` class provided by `fastai`. The fastai library also implements various transforms for data augmentation to improve training. While there are some defaults that I leave intact, I add vertical flipping (`do_flip=True`) as this has been commonly used for this particular problem.
# 
# Typically, one would use the `ImageDataBunch` class to load the dataset much easier, but as I will adjust the splitting and add oversampling in future kernels, I have used this customized creation of the DataBunch using the `data_block` API.

# In[ ]:


tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.0,max_lighting=0.1,p_lighting=0.5)
src = (ImageList.from_df(df=df,path='./',cols='path') #get dataset from dataset
        .split_by_idx(range(len(train_df)-1,len(df))) #Splitting the dataset
        .label_from_df(cols='level') #obtain labels from the level column
      )
data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') #Data augmentation
        .databunch(bs=bs,num_workers=4) #DataBunch
        .normalize(imagenet_stats) #Normalize     
       )


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# # Training (Transfer learning) 

# **Training:**
# 
# We use transfer learning, where we retrain the last layers of a pretrained neural network. I use the ResNet50 architecture trained on the ImageNet dataset, which has been commonly used for pre-training applications in computer vision. Fastai makes it quite simple to create a model and train:

# In[ ]:


import torchvision
from fastai.metrics import *
from fastai.callbacks import *
learn = cnn_learner(data, models.resnet50, wd = 1e-5, metrics = [accuracy,AUROC()],callback_fns=[partial(CSVLogger,append=True)])


# We use the learning-rate finder developed by Dr. Leslie Smith and implemented by the fastai team in their library:

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)


# Here we can see that the loss decreases fastest around `lr=1e-2` so that is what we will use to train: 

# In[ ]:


learn.fit_one_cycle(1,max_lr = 1e-2)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-1-512')


# The previous model only trained the last model. We can unfreeze the rest of the model, and train the rest of the model using discriminative learning rates. The first layers aren't changed as much, with lower learning rates, while the last layers are changed more, with higher learning rates. We use the learning rate finder again, and use a range of learning rates for different layers in the neural network.

# In[ ]:


learn.load('stage-1-512')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(16, max_lr=slice(1e-6,1e-3))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


learn.save('stage-2-512')


# # Checking results

# We look at our predictions and make a confusion matrix.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


from sklearn.metrics import roc_auc_score, confusion_matrix
def auc_score(y_score,y_true):
    try:
      auc = torch.tensor(roc_auc_score(y_true,y_score[:,1]))
    except:
      auc = torch.tensor(np.nan)
    return auc

probs,val_labels = learn.get_preds(ds_type=DatasetType.Valid) 
print('Accuracy',accuracy(probs,val_labels)),
print('Error Rate', error_rate(probs, val_labels))
print('AUC', auc_score(probs,val_labels))


tn, fp, fn, tp = confusion_matrix(val_labels,(probs[:,1]>0.5).float()).ravel()

specificity = tn/(tn+fp)
sensitivity = tp/(tp+fn)

print('Specificity and Sensitivity',specificity,sensitivity)


# Our model has decent AUC (around 0.92), which is similar to some of the [results](https://arxiv.org/abs/1803.04337) demonstrated previously. However, this is still not close to AUC=0.99 demonstrated by Google.
# 
# We can use an optimized threshold as well:

# In[ ]:


from sklearn.metrics import roc_curve
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 
  

print(probs[:,1]) 

threshold = Find_Optimal_Cutoff(val_labels,probs[:,1])
print(threshold)

print(confusion_matrix(val_labels,probs[:,1]>threshold[0]))

tn, fp, fn, tp = confusion_matrix(val_labels,probs[:,1]>threshold[0]).ravel()

specificity = tn/(tn+fp)
sensitivity = tp/(tp+fn)

print('Specificity and Sensitivity',specificity,sensitivity)


# We will now export our model so it can be used for inference.

# In[ ]:


learn.export()

