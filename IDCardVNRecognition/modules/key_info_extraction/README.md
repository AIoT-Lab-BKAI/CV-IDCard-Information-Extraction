# key_information_extraction
## F1 score on validation dataset

| **name**            |  **mEP** |  **mER** |  **mEF** |   **mEA**|
| -------------------- | --------- | -------- | -------- | ------- |
| exchange_rate_val   | 0        | 0        | 0        | 0        |
| company_name_val    | 0.985507 | 0.957746 | 0.971429 | 0.957746 |
| serial              | 1        | 1        | 1        | 1        |
| website             | 0        | 0        | 0        | 0        |
| amount_in_words     | 1        | 1        | 1        | 1        |
| address_val         | 0.93     | 0.978947 | 0.953846 | 0.978947 |
| buyer               | 0.916667 | 0.846154 | 0.88     | 0.846154 |
| no_val              | 0.942857 | 1        | 0.970588 | 1        |
| date                | 1        | 1        | 1        | 1        |
| VAT_rate            | 0.964286 | 0.964286 | 0.964286 | 0.964286 |
| tax_code_val        | 1        | 1        | 1        | 1        |
| address             | 0.927536 | 0.969697 | 0.948148 | 0.969697 |
| amount_in_words_val | 0.933333 | 0.933333 | 0.933333 | 0.933333 |
| no                  | 0.972222 | 1        | 0.985915 | 1        |
| company_name        | 0.944444 | 0.918919 | 0.931507 | 0.918919 |
| seller              | 0.3      | 0.230769 | 0.26087  | 0.230769 |
| tax_code            | 1        | 1        | 1        | 1        |
| total_val           | 0.939394 | 0.96875  | 0.953846 | 0.96875  |
| bank                | 0        | 0        | 0        | 0        |
| VAT_rate_val        | 0.925926 | 0.961538 | 0.943396 | 0.961538 |
| form_val            | 1        | 0.969697 | 0.984615 | 0.969697 |
| account_no          | 0.25     | 0.153846 | 0.190476 | 0.153846 |
| serial_val          | 0.970588 | 1        | 0.985075 | 1        |
| grand_total_val     | 1        | 1        | 1        | 1        |
| total               | 1        | 0.969697 | 0.984615 | 0.969697 |
| exchange_rate       | 1        | 1        | 1        | 1        |
| grand_total         | 1        | 1        | 1        | 1        |
| VAT_amount_val      | 1        | 0.923077 | 0.96     | 0.923077 |
| VAT_amount          | 0.962963 | 0.962963 | 0.962963 | 0.962963 |
| form                | 1        | 1        | 1        | 1        |
| **Overall**             | 0.951904 | 0.951904 | 0.951904 | 0.951904 |

## Speed Processing 
Document: max 100 lines, max characeters in 1 line is 100 character
- GPU : 0.26s /doc
- CPU : 2.5s /doc

## Result Key extraction sample

| **Key**            |  **Value** |
|-------------------- |---------|
|form|M???u s??? (Form no):|
|form_val|07KPTQ0/001|
|serial_val|CD/19E|
|serial|K?? hi???u (Sign):|
|no|S??? (No):|
|no_val|0000141|
|date|Ng??y (Date) 15 th??ng (month) 3 n??m (year)  2020|
|company_name|????n v??? b??n h??ng (Supplier):|
|company_name_val|C??NG TY TNHH CDL PRECISION TECHNOLOGY (VIETNAM)|
|tax_code|M?? s??? thu??? (Tax code):|
|tax_code_val|2500546489|
|address_val|L?? 14, Khu c??ng nghi???p B??nh Xuy??n, X?? S??n L??i, Huy???n B??nh Xuy??n, T???nh V??nh\|
|address|?????a ch??? (Address):|
|address_val|Ph??c , Vi???t Nam|
|buyer|H??? t??n ng?????i mua h??ng (Buyer):|
|company_name|T??n ????n v??? (Company's name):|
|company_name_val|C??ng ty TNHH Samsung Display Vi???t Nam|
|tax_code_val|2300852009|
|tax_code|M?? s??? thu??? (Tax code):|
|address_val|Khu C??ng Nghi???p Y??n Phong, X?? Y??n Trung, Huy???n Y??n Phong, T???nh B???c Ninh, Vi???t|
|address|?????a ch??? (Address):|
|address_val|Nam|
|account_no|S??? t??i kho???n (Bank account)| 
|total_val|144,0000|
|total|C???ng ti???n b??n h??ng h??a, d???ch v??? (Total amount):|
|amount_in_words|S??? ti???n vi???t b???ng ch??? (Total amount in words): |
|amount_in_words_val|M???t tr??m b???n m????i b???n ???? la M??? ch???n.|
|exchange_rate_val|23.105 VND/USD|
|exchange_rate|T??? gi?? (Exchange rate):|
|date|Ng??y chuy???n ?????i (conversion date): 15/03/2020|


