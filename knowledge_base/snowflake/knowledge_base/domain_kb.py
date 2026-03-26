"""
DataPilot – Domain Knowledge Bases
Production-grade industry context based on real-world data warehouse
and star schema conventions used by enterprise analytics teams.

Each domain context is written to match dimensional modeling naming
conventions (DIM_/FACT_ prefixes) since that is the industry standard
for production data warehouses across all verticals.

The context is injected into the Llama prompt before schema analysis
so the model starts from an accurate understanding of the business
rather than guessing from generic table names.
"""
from __future__ import annotations

DOMAINS: list[str] = [
    "Banking & Finance",
    "Manufacturing",
    "Healthcare & Compounding Pharmacy",
    "Retail & E-commerce",
    "SaaS & Technology",
    "Logistics & Supply Chain",
    "Other",
]

DOMAIN_CONTEXT: dict[str, str] = {

# ══════════════════════════════════════════════════════════════════════════════
"Banking & Finance": """
INDUSTRY: Banking & Financial Services
SCHEMA CONVENTION: Star schema with DIM_ and FACT_ prefix naming.

COMMON TABLE NAMES:
  Fact tables   : FACT_Transaction, FACT_Loan_Account, FACT_Account_Balance,
                  FACT_Trade, FACT_Payment, FACT_GL_Entry, FACT_Fee_Revenue
  Dim tables    : DIM_Customer, DIM_Account, DIM_Branch, DIM_Employee,
                  DIM_Product, DIM_Channel, DIM_Geography, DIM_Date,
                  DIM_Loan_Type, DIM_Collateral, DIM_Risk_Rating,
                  DIM_Currency, DIM_Counterparty, DIM_Cost_Center

KEY METRICS AND EXACT SQL:
  Total deposits           : SUM(Total_Deposits_USD) from FACT_Account_Balance
  Total withdrawals        : SUM(Total_Withdrawals_USD) from FACT_Account_Balance
  Closing balance (AUM)    : SUM(Closing_Balance_USD) from FACT_Account_Balance
  Average daily balance    : AVG(Average_Daily_Balance_USD) from FACT_Account_Balance
  Transaction volume       : SUM(Transaction_Amount_USD) from FACT_Transaction
                             WHERE Transaction_Status = 'Settled'
  Transaction count        : COUNT(Transaction_ID) from FACT_Transaction
                             WHERE Transaction_Status = 'Settled'
  Fee income               : SUM(Fee_Amount_USD) from FACT_Transaction
  Total loans outstanding  : SUM(Outstanding_Balance_USD) from FACT_Loan_Account
                             WHERE Loan_Status = 'Active'
  NPL (Non-Performing)     : SUM(Outstanding_Balance_USD) from FACT_Loan_Account
                             WHERE Loan_Status = 'Non_Performing' OR Days_Past_Due > 90
  ECL (Expected Credit Loss): SUM(Expected_Credit_Loss_USD) from FACT_Loan_Account
  Interest income          : SUM(Interest_Paid_USD) from FACT_Loan_Account
  Principal collected      : SUM(Principal_Paid_USD) from FACT_Loan_Account
  Accrued interest         : SUM(Accrued_Interest_This_Month_USD) from FACT_Loan_Account
  LTV ratio                : AVG(LTV_Ratio) from FACT_Loan_Account
  DTI ratio                : AVG(DTI_Ratio) from FACT_Loan_Account
  Customer assets (AUM)    : SUM(Total_Assets_With_Bank_USD) from DIM_Customer
  Suspicious transactions  : COUNT WHERE Suspicious_Activity_Flag = True
  Large currency txns      : COUNT WHERE Large_Currency_Transaction_Flag = True

BUSINESS VOCABULARY:
  AUM         : Assets Under Management — Total_Assets_With_Bank_USD in DIM_Customer
  CASA        : Current Account Savings Account — Account_Type IN ('Checking','Savings')
  NIM         : Net Interest Margin — interest income minus interest expense divided by earning assets
  NPL / NPA   : Non-Performing Loan/Asset — Days_Past_Due > 90 or Loan_Status = 'Non_Performing'
  ECL         : Expected Credit Loss — provisioned loss on loan portfolio
  LTV         : Loan-to-Value ratio — Outstanding_Balance divided by Collateral_Value
  DTI         : Debt-to-Income ratio — monthly debt payments divided by gross monthly income
  CIF         : Customer Information File — unique customer ID (CIF_Number)
  KYC         : Know Your Customer — KYC_Status column (Verified / Pending / Failed)
  AML         : Anti-Money Laundering — AML_Flag in DIM_Customer
  PEP         : Politically Exposed Person — Is_PEP flag in DIM_Customer
  Credit tier : Customer risk band — Credit_Tier column (Prime / Near-Prime / Sub-Prime)
  Delinquency : Days_Past_Due and Delinquency_Bucket in FACT_Loan_Account
  Restructured: Loan that was modified — Restructured flag in FACT_Loan_Account
  Channel     : How transaction was initiated — DIM_Channel (Branch / Mobile / ATM / Online)
  FX          : Foreign exchange — FX_Rate, Foreign_Currency, USD_Equivalent_Amount

COMMON FILTERS (always apply unless user asks otherwise):
  Active accounts       : Status = 'Active'
  Settled transactions  : Transaction_Status = 'Settled'
  Active loans          : Loan_Status = 'Active'
  Exclude PEP / AML     : Is_PEP = False AND AML_Flag = False (for standard customer metrics)

STANDARD FK RELATIONSHIPS (star schema joins):
  FACT_Transaction.Account_ID           -> DIM_Account.Account_ID
  FACT_Transaction.Customer_ID          -> DIM_Customer.Customer_ID
  FACT_Transaction.Channel_ID           -> DIM_Channel.Channel_ID
  FACT_Transaction.Branch_ID            -> DIM_Branch.Branch_ID
  FACT_Transaction.Employee_ID          -> DIM_Employee.Employee_ID
  FACT_Transaction.Transaction_Date_ID  -> DIM_Date.Date_ID
  FACT_Loan_Account.Customer_ID         -> DIM_Customer.Customer_ID
  FACT_Loan_Account.Account_ID          -> DIM_Account.Account_ID
  FACT_Loan_Account.Loan_Type_ID        -> DIM_Loan_Type.Loan_Type_ID
  FACT_Loan_Account.Collateral_ID       -> DIM_Collateral.Collateral_ID
  FACT_Loan_Account.Risk_Rating_ID      -> DIM_Risk_Rating.Risk_Rating_ID
  FACT_Loan_Account.Branch_ID           -> DIM_Branch.Branch_ID
  FACT_Loan_Account.Origination_Date_ID -> DIM_Date.Date_ID
  FACT_Account_Balance.Account_ID       -> DIM_Account.Account_ID
  FACT_Account_Balance.Customer_ID      -> DIM_Customer.Customer_ID
  FACT_Account_Balance.Branch_ID        -> DIM_Branch.Branch_ID
  FACT_Account_Balance.Geography_ID     -> DIM_Geography.Geography_ID
  FACT_Account_Balance.Snapshot_Date_ID -> DIM_Date.Date_ID
  DIM_Account.Customer_ID               -> DIM_Customer.Customer_ID
  DIM_Account.Branch_ID                 -> DIM_Branch.Branch_ID
  DIM_Branch.Geography_ID               -> DIM_Geography.Geography_ID
""",

# ══════════════════════════════════════════════════════════════════════════════
"Manufacturing": """
INDUSTRY: Manufacturing & Industrial Production
SCHEMA CONVENTION: Star schema with DIM_ and FACT_ prefix naming.

COMMON TABLE NAMES:
  Fact tables   : FACT_Production_Order, FACT_Work_Order, FACT_Quality_Check,
                  FACT_Inventory_Movement, FACT_Machine_Downtime,
                  FACT_Purchase_Order, FACT_Sales_Order, FACT_Scrap,
                  FACT_Maintenance_Event, FACT_Shift_Log
  Dim tables    : DIM_Product, DIM_Machine, DIM_Employee, DIM_Shift,
                  DIM_Supplier, DIM_Customer, DIM_Plant, DIM_Work_Center,
                  DIM_BOM, DIM_Date, DIM_Material, DIM_Defect_Type,
                  DIM_Downtime_Reason, DIM_Cost_Center

KEY METRICS AND EXACT SQL:
  Total units produced     : SUM(Actual_Qty) from FACT_Production_Order
                             WHERE Order_Status = 'Completed'
  Good units (yield)       : SUM(Released_Qty) from FACT_Production_Order
  Rejected units           : SUM(Rejected_Qty) from FACT_Production_Order
  Yield rate               : SUM(Released_Qty) / SUM(Planned_Qty) * 100
  Defect rate              : SUM(Rejected_Qty) / SUM(Actual_Qty) * 100
  Scrap qty                : SUM(Scrap_Qty) from FACT_Scrap
  Scrap rate               : SUM(Scrap_Qty) / SUM(Actual_Qty) * 100
  OEE                      : Availability * Performance * Quality (0-1 scale each)
  Avg cycle time           : AVG(Actual_Prep_Min) from FACT_Production_Order
  Machine downtime hrs     : SUM(Downtime_Duration_Min) / 60 from FACT_Machine_Downtime
  MTBF                     : Total uptime hours / COUNT(breakdown events)
  MTTR                     : AVG(Repair_Duration_Min) from FACT_Maintenance_Event
  Total production cost    : SUM(Total_Batch_Cost_USD) from FACT_Production_Order
  Labor cost               : SUM(Labor_Cost_USD) from FACT_Production_Order
  Material cost            : SUM(Material_Cost_USD) from FACT_Production_Order
  Inventory value          : SUM(Qty_On_Hand * Unit_Cost_USD) from FACT_Inventory
  On-time delivery rate    : COUNT(delivered on time) / COUNT(total orders) * 100
  Purchase order value     : SUM(PO_Amount_USD) from FACT_Purchase_Order

BUSINESS VOCABULARY:
  BOM         : Bill of Materials — component list for a product (DIM_BOM)
  WIP         : Work In Progress — units started but not yet finished
  MRP         : Material Requirements Planning — demand-driven ordering system
  SKU         : Stock Keeping Unit — unique product variant ID
  OEE         : Overall Equipment Effectiveness — availability x performance x quality
  MTBF        : Mean Time Between Failures — avg operating hours between breakdowns
  MTTR        : Mean Time To Repair — avg hours to restore machine after failure
  Cycle time  : Time to complete one production unit (Actual_Prep_Min)
  Takt time   : Available production time divided by customer demand rate
  Planned qty : Target production quantity (Planned_Qty)
  Actual qty  : Units actually produced (Actual_Qty)
  Released qty: Units approved after quality control (Released_Qty)
  Rejected qty: Units that failed quality control (Rejected_Qty)
  Scrap       : Unusable material or units that cannot be reworked
  Rework      : Units that failed QC but can be repaired and reused
  Yield       : Released_Qty / Planned_Qty — production efficiency ratio
  Lead time   : Time from order placement to delivery
  CAPA        : Corrective and Preventive Action — response to deviation
  Deviation   : Departure from standard procedure (Deviation_Reported flag)

COMMON FILTERS:
  Completed orders only  : Order_Status = 'Completed' or Batch_Status = 'Released'
  Exclude cancelled      : Order_Status != 'Cancelled'
  Active machines        : Machine_Status = 'Active'
  Active suppliers       : Supplier_Status = 'Active'

STANDARD FK RELATIONSHIPS:
  FACT_Production_Order.Product_ID    -> DIM_Product.Product_ID
  FACT_Production_Order.Machine_ID    -> DIM_Machine.Machine_ID
  FACT_Production_Order.Employee_ID   -> DIM_Employee.Employee_ID
  FACT_Production_Order.Date_ID       -> DIM_Date.Date_ID
  FACT_Production_Order.Plant_ID      -> DIM_Plant.Plant_ID
  FACT_Work_Order.Production_Order_ID -> FACT_Production_Order.Order_ID
  FACT_Quality_Check.Order_ID         -> FACT_Production_Order.Order_ID
  FACT_Machine_Downtime.Machine_ID    -> DIM_Machine.Machine_ID
  FACT_Inventory_Movement.Product_ID  -> DIM_Product.Product_ID
  FACT_Purchase_Order.Supplier_ID     -> DIM_Supplier.Supplier_ID
""",

# ══════════════════════════════════════════════════════════════════════════════
"Healthcare & Compounding Pharmacy": """
INDUSTRY: Healthcare, Hospitals, Clinics, and Compounding Pharmacy
SCHEMA CONVENTION: Star schema with DIM_ and FACT_ prefix naming.

SECTION A — GENERAL HEALTHCARE (Hospitals and Clinics)

COMMON TABLE NAMES (Hospital):
  Fact tables   : FACT_Encounter, FACT_Claim, FACT_Procedure, FACT_Lab_Result,
                  FACT_Medication_Order, FACT_Admission, FACT_Revenue_Cycle,
                  FACT_Readmission, FACT_ED_Visit
  Dim tables    : DIM_Patient, DIM_Provider, DIM_Facility, DIM_Payer,
                  DIM_Diagnosis, DIM_Procedure, DIM_Medication,
                  DIM_Department, DIM_Date, DIM_Geography

KEY METRICS (Hospital):
  Patient visits           : COUNT(DISTINCT Encounter_ID) from FACT_Encounter
  Unique patients          : COUNT(DISTINCT Patient_ID) from FACT_Encounter
  Revenue                  : SUM(Total_Charge_USD) from FACT_Claim
                             WHERE Claim_Status = 'Paid'
  Net revenue              : SUM(Net_Revenue_USD) from FACT_Revenue_Cycle
  Average LOS              : AVG(DATEDIFF(day, Admission_Date, Discharge_Date))
  Readmission rate         : patients readmitted within 30 days / total discharged
  Claim denial rate        : COUNT(denied) / COUNT(submitted) * 100
  Outstanding AR           : SUM(Balance_Due_USD) from FACT_Revenue_Cycle
                             WHERE Payment_Status = 'Pending'

BUSINESS VOCABULARY (Hospital):
  LOS         : Length of Stay — Discharge_Date minus Admission_Date in days
  ICD codes   : International Classification of Diseases — diagnosis codes
  CPT codes   : Current Procedural Terminology — procedure billing codes
  OPD         : Outpatient Department — Encounter_Type = 'Outpatient'
  IPD         : Inpatient Department — Encounter_Type = 'Inpatient'
  Readmission : Return visit within 30 days of discharge
  Co-pay      : Patient cost share (Copay_USD)
  Prior auth  : Insurance pre-approval (Prior_Auth_Obtained flag)
  AR days     : Average days outstanding on receivables

SECTION B — COMPOUNDING PHARMACY

COMMON TABLE NAMES (Compounding Pharmacy):
  Fact tables   : FACT_Prescription_Fill, FACT_Compounding_Batch,
                  FACT_Ingredient_Usage, FACT_QC_Test
  Dim tables    : DIM_Patient, DIM_Prescriber, DIM_Formula, DIM_Ingredient,
                  DIM_Supplier, DIM_Staff, DIM_Payer, DIM_Equipment,
                  DIM_Date, DIM_Geography

KEY METRICS (Compounding Pharmacy):
  Prescription fills       : COUNT(Rx_Fill_ID) from FACT_Prescription_Fill
                             WHERE Fill_Status = 'Dispensed'
  Total revenue            : SUM(Total_Charge_USD) from FACT_Prescription_Fill
                             WHERE Fill_Status = 'Dispensed'
  Gross profit             : SUM(Gross_Profit_USD) from FACT_Prescription_Fill
                             WHERE Fill_Status = 'Dispensed'
  Profit margin            : AVG(Margin_Pct) from FACT_Prescription_Fill
  COGS                     : SUM(COGS_USD) from FACT_Prescription_Fill
  Insurance collections    : SUM(Insurance_Payment_USD) from FACT_Prescription_Fill
  Patient copay            : SUM(Copay_USD) from FACT_Prescription_Fill
  Patient payment          : SUM(Patient_Payment_USD) from FACT_Prescription_Fill
  Batch yield              : AVG(Yield_Pct) from FACT_Compounding_Batch
                             WHERE Batch_Status = 'Released'
  Batch waste              : AVG(Waste_Pct) from FACT_Compounding_Batch
  Rejected batches         : COUNT(Batch_ID) from FACT_Compounding_Batch
                             WHERE Batch_Status = 'Rejected'
  Total batch cost         : SUM(Total_Batch_Cost_USD) from FACT_Compounding_Batch
  Labor cost               : SUM(Labor_Cost_USD) from FACT_Compounding_Batch
  Ingredient cost          : SUM(Ingredient_Cost_USD) from FACT_Compounding_Batch
  Sterility pass rate      : COUNT WHERE Sterility_Test_Result = 'Pass'
                             / COUNT WHERE Sterility_Test_Performed = True * 100
  QC pass rate             : COUNT WHERE QC_Visual_Inspection = 'Pass'
                             / COUNT(Batch_ID) * 100
  Rx count by doctor       : COUNT(Rx_Fill_ID) GROUP BY Prescriber_ID
  Rx count by formula      : COUNT(Rx_Fill_ID) GROUP BY Formula_ID

BUSINESS VOCABULARY (Compounding Pharmacy):
  Compounding : Custom preparation of medication not commercially available
  Formula     : Master compound recipe definition (DIM_Formula)
  MFR         : Master Formula Record number — MFR_Number in DIM_Formula
  BUD         : Beyond Use Date — stability window for compounded med (BUD_Days)
  USP         : United States Pharmacopeia quality standard
  USP 797     : Standard for sterile compounding — USP797_Trained flag in DIM_Staff
  USP 800     : Standard for hazardous drug handling — Is_Hazardous_USP800, USP800_Trained
  QC          : Quality Control — QC_Visual_Inspection, Sterility_Test_Result, Potency_Test_Result
  Sterile     : Requires sterile prep environment — Is_Sterile in DIM_Formula
  Hazardous   : Requires special handling — Is_Hazardous_USP800 in DIM_Formula
  CAPA        : Corrective and Preventive Action — CAPA_Required flag
  Batch       : One production run of a formula (FACT_Compounding_Batch)
  Lot number  : Unique batch tracking ID (Lot_Number)
  Yield pct   : Percent of batch successfully produced (Yield_Pct)
  Waste pct   : Percent of batch discarded (Waste_Pct)
  Prior auth  : Insurance pre-approval — Prior_Auth_Obtained flag
  Dispensing  : Giving medication to patient — Dispensing_Method, Dispensing_Staff_ID
  Prescriber  : Doctor who wrote the prescription — DIM_Prescriber
  Payer       : Insurance company — DIM_Payer with Payer_Type, Collection_Rate
  Rx type     : New, Refill, or Transfer prescription (Rx_Type)
  Refill      : Repeat fill (Refill_Number > 0)
  Adverse     : Adverse drug reaction (Adverse_Reaction flag)
  DEA         : Drug Enforcement Administration number for controlled substances
  NPI         : National Provider Identifier — NPI_Number in DIM_Prescriber

COMMON FILTERS (Compounding Pharmacy):
  Dispensed Rx only    : Fill_Status = 'Dispensed'
  Released batches     : Batch_Status = 'Released'
  Active prescribers   : Status = 'Active' in DIM_Prescriber
  Active patients      : Status = 'Active' in DIM_Patient
  New prescriptions    : Rx_Type = 'New' or Refill_Number = 0
  Sterile compounds    : Is_Sterile = True in DIM_Formula
  Hazardous compounds  : Is_Hazardous_USP800 = True in DIM_Formula

STANDARD FK RELATIONSHIPS (Compounding Pharmacy):
  FACT_Prescription_Fill.Patient_ID          -> DIM_Patient.Patient_ID
  FACT_Prescription_Fill.Prescriber_ID       -> DIM_Prescriber.Prescriber_ID
  FACT_Prescription_Fill.Formula_ID          -> DIM_Formula.Formula_ID
  FACT_Prescription_Fill.Payer_ID            -> DIM_Payer.Payer_ID
  FACT_Prescription_Fill.Dispensing_Staff_ID -> DIM_Staff.Staff_ID
  FACT_Prescription_Fill.Fill_Date_ID        -> DIM_Date.Date_ID
  FACT_Compounding_Batch.Formula_ID          -> DIM_Formula.Formula_ID
  FACT_Compounding_Batch.Supplier_ID         -> DIM_Supplier.Supplier_ID
  FACT_Compounding_Batch.Compounding_Staff_ID-> DIM_Staff.Staff_ID
  FACT_Compounding_Batch.QC_Staff_ID         -> DIM_Staff.Staff_ID
  FACT_Compounding_Batch.Equipment_ID        -> DIM_Equipment.Equipment_ID
  FACT_Compounding_Batch.Batch_Date_ID       -> DIM_Date.Date_ID
  FACT_Compounding_Batch.Release_Date_ID     -> DIM_Date.Date_ID
""",

# ══════════════════════════════════════════════════════════════════════════════
"Retail & E-commerce": """
INDUSTRY: Retail, E-commerce, and Omnichannel Commerce
SCHEMA CONVENTION: Star schema with DIM_ and FACT_ prefix naming.

COMMON TABLE NAMES:
  Fact tables   : FACT_Order, FACT_Order_Item, FACT_Return, FACT_Inventory,
                  FACT_Web_Session, FACT_Cart_Abandonment, FACT_Promotion_Usage,
                  FACT_Shipment, FACT_Review, FACT_Customer_Event
  Dim tables    : DIM_Customer, DIM_Product, DIM_Category, DIM_Supplier,
                  DIM_Store, DIM_Channel, DIM_Promotion, DIM_Geography,
                  DIM_Date, DIM_Carrier, DIM_Employee

KEY METRICS AND EXACT SQL:
  Gross revenue (GMV)      : SUM(Order_Total_USD) from FACT_Order
                             WHERE Order_Status NOT IN ('Cancelled','Fraud')
  Net revenue              : SUM(Order_Total_USD - Discount_Amount_USD - Refund_Amount_USD)
                             from FACT_Order WHERE Order_Status = 'Delivered'
  Total orders             : COUNT(DISTINCT Order_ID) from FACT_Order
                             WHERE Order_Status NOT IN ('Cancelled','Fraud')
  Units sold               : SUM(Qty_Ordered) from FACT_Order_Item
  AOV (avg order value)    : SUM(Order_Total_USD) / COUNT(DISTINCT Order_ID)
  Return rate              : COUNT(Return_ID) / COUNT(Order_ID) * 100
  Return value             : SUM(Refund_Amount_USD) from FACT_Return
  Gross profit             : SUM(Gross_Profit_USD) from FACT_Order_Item
  Gross margin pct         : SUM(Gross_Profit_USD) / SUM(Revenue_USD) * 100
  Unique customers         : COUNT(DISTINCT Customer_ID) from FACT_Order
  New customers            : COUNT(DISTINCT Customer_ID) WHERE Is_First_Order = True
  LTV (lifetime value)     : SUM(Order_Total_USD) GROUP BY Customer_ID
  Inventory on hand        : SUM(Qty_On_Hand) from FACT_Inventory
  Cart abandonment rate    : COUNT(abandoned) / COUNT(total carts) * 100
  Conversion rate          : COUNT(orders) / COUNT(sessions) * 100
  On-time delivery rate    : COUNT WHERE Actual_Delivery_Date <= Promised_Delivery_Date

BUSINESS VOCABULARY:
  GMV         : Gross Merchandise Value — total order value before deductions
  AOV         : Average Order Value — revenue divided by order count
  CAC         : Customer Acquisition Cost — marketing spend divided by new customers
  LTV         : Lifetime Value — total revenue from one customer over their life
  SKU         : Stock Keeping Unit — unique product variant (Size/Color/Style)
  Churn       : Customers who have not purchased in a defined period
  NPS         : Net Promoter Score — customer satisfaction metric
  Markdown    : Price reduction — Discount_Amount_USD or Discount_Pct
  BOPIS       : Buy Online Pick In Store — Fulfillment_Type = 'BOPIS'
  Shrinkage   : Inventory loss due to theft, damage, or error
  Sell-through: Units sold divided by units received — inventory performance
  First order : Customer's first ever purchase (Is_First_Order flag)
  Repeat rate : Percentage of customers who purchase more than once
  Basket size : Number of items per order (Qty_Ordered per Order_ID)
  ROAS        : Return on Ad Spend — revenue divided by advertising cost

COMMON FILTERS:
  Valid orders only    : Order_Status NOT IN ('Cancelled', 'Fraud', 'Test')
  Delivered orders     : Order_Status = 'Delivered'
  Exclude internal     : Customer_Type != 'Employee' or Is_Internal = False
  Active products      : Product_Status = 'Active'
  Active stores        : Store_Status = 'Active'

STANDARD FK RELATIONSHIPS:
  FACT_Order.Customer_ID       -> DIM_Customer.Customer_ID
  FACT_Order.Store_ID          -> DIM_Store.Store_ID
  FACT_Order.Channel_ID        -> DIM_Channel.Channel_ID
  FACT_Order.Order_Date_ID     -> DIM_Date.Date_ID
  FACT_Order.Geography_ID      -> DIM_Geography.Geography_ID
  FACT_Order_Item.Order_ID     -> FACT_Order.Order_ID
  FACT_Order_Item.Product_ID   -> DIM_Product.Product_ID
  FACT_Order_Item.Promotion_ID -> DIM_Promotion.Promotion_ID
  FACT_Return.Order_ID         -> FACT_Order.Order_ID
  FACT_Inventory.Product_ID    -> DIM_Product.Product_ID
  FACT_Inventory.Store_ID      -> DIM_Store.Store_ID
  FACT_Shipment.Order_ID       -> FACT_Order.Order_ID
  FACT_Shipment.Carrier_ID     -> DIM_Carrier.Carrier_ID
  DIM_Product.Category_ID      -> DIM_Category.Category_ID
  DIM_Product.Supplier_ID      -> DIM_Supplier.Supplier_ID
""",

# ══════════════════════════════════════════════════════════════════════════════
"SaaS & Technology": """
INDUSTRY: SaaS, Software, and Technology Products
SCHEMA CONVENTION: Star schema with DIM_ and FACT_ prefix naming.
Also common: event-based schemas using tables like events, sessions,
users, subscriptions, invoices, feature_usage without strict DIM/FACT prefixes.

COMMON TABLE NAMES:
  Fact tables   : FACT_Subscription, FACT_Invoice, FACT_Event, FACT_Session,
                  FACT_Feature_Usage, FACT_Support_Ticket, FACT_Churn_Event,
                  FACT_Expansion, FACT_Conversion, FACT_API_Call
  Dim tables    : DIM_Account, DIM_User, DIM_Plan, DIM_Feature,
                  DIM_Geography, DIM_Channel, DIM_Date, DIM_Industry

KEY METRICS AND EXACT SQL:
  MRR                     : SUM(Monthly_Amount_USD) from FACT_Subscription
                            WHERE Status = 'Active'
  ARR                     : SUM(Monthly_Amount_USD) * 12 from FACT_Subscription
                            WHERE Status = 'Active'
  New MRR                 : SUM(Monthly_Amount_USD) WHERE Is_New_Customer = True
  Expansion MRR           : SUM(Expansion_Amount_USD) from FACT_Expansion
  Churned MRR             : SUM(Monthly_Amount_USD) from FACT_Churn_Event
  Net MRR growth          : (New + Expansion) - (Churn + Contraction)
  NRR                     : (Start MRR + Expansion - Churn - Contraction) / Start MRR
  GRR                     : (Start MRR - Churn - Contraction) / Start MRR
  Total revenue           : SUM(Invoice_Amount_USD) from FACT_Invoice
                            WHERE Payment_Status = 'Paid'
  ARPA                    : ARR / COUNT(DISTINCT Account_ID)
  Active customers        : COUNT(DISTINCT Account_ID) from FACT_Subscription
                            WHERE Status = 'Active'
  Monthly churn rate      : Churned accounts / Start-of-month accounts * 100
  Annual churn rate       : Churned accounts / Start-of-year accounts * 100
  DAU                     : COUNT(DISTINCT User_ID) WHERE DATE(Event_Date) = today
  MAU                     : COUNT(DISTINCT User_ID) WHERE Event_Date >= month start
  DAU to MAU ratio        : DAU / MAU — engagement stickiness metric
  Feature adoption        : COUNT(DISTINCT users using feature) / total active users
  Trial conversion rate   : COUNT(converted) / COUNT(trials) * 100
  Avg resolution time     : AVG(Resolution_Time_Hours) from FACT_Support_Ticket
  Payback period months   : CAC / (ARPA * Gross_Margin_Pct)

BUSINESS VOCABULARY:
  MRR         : Monthly Recurring Revenue — active subscription revenue per month
  ARR         : Annual Recurring Revenue — MRR multiplied by 12
  Churn       : Cancelled subscriptions — Status changed to 'Cancelled'
  Contraction : Downgrade — revenue decrease from existing customer
  Expansion   : Upsell or upgrade — revenue increase from existing customer
  NRR         : Net Revenue Retention — includes expansion and churn
  GRR         : Gross Revenue Retention — churn only, no expansion credit
  ARPA        : Average Revenue Per Account
  CAC         : Customer Acquisition Cost — fully loaded cost per new customer
  LTV         : Lifetime Value — total expected revenue from one customer
  DAU / MAU   : Daily / Monthly Active Users — engagement metrics
  Activation  : User completing the key first-time setup action
  Cohort      : Group of users who started in the same time period
  Freemium    : Free tier — Plan_Type = 'Free', not counted in MRR
  Trial       : Free trial period — Is_Trial = True or Plan_Type = 'Trial'
  Seat        : Individual user licence — Seat_Count in FACT_Subscription
  Upsell      : Adding seats or upgrading plan within same account
  Booking     : New contract value signed, not yet revenue
  TCV         : Total Contract Value — full value of multi-year deal

COMMON FILTERS:
  Active subscriptions   : Status = 'Active'
  Paying customers only  : Plan_Type NOT IN ('Free','Trial')
  Exclude internal users : Is_Internal = False or Is_Test = False
  Exclude bot events     : Event_Type != 'bot' or Is_Bot = False
  Exclude churned        : Status != 'Cancelled' for active metrics

STANDARD FK RELATIONSHIPS:
  FACT_Subscription.Account_ID   -> DIM_Account.Account_ID
  FACT_Subscription.Plan_ID      -> DIM_Plan.Plan_ID
  FACT_Subscription.Date_ID      -> DIM_Date.Date_ID
  FACT_Invoice.Account_ID        -> DIM_Account.Account_ID
  FACT_Invoice.Subscription_ID   -> FACT_Subscription.Subscription_ID
  FACT_Event.User_ID             -> DIM_User.User_ID
  FACT_Event.Account_ID          -> DIM_Account.Account_ID
  FACT_Event.Feature_ID          -> DIM_Feature.Feature_ID
  FACT_Session.User_ID           -> DIM_User.User_ID
  FACT_Support_Ticket.Account_ID -> DIM_Account.Account_ID
  FACT_Churn_Event.Account_ID    -> DIM_Account.Account_ID
  DIM_User.Account_ID            -> DIM_Account.Account_ID
""",

# ══════════════════════════════════════════════════════════════════════════════
"Logistics & Supply Chain": """
INDUSTRY: Logistics, Transportation, and Supply Chain Management
SCHEMA CONVENTION: Star schema with DIM_ and FACT_ prefix naming.

COMMON TABLE NAMES:
  Fact tables   : FACT_Shipment, FACT_Delivery_Event, FACT_Route,
                  FACT_Warehouse_Movement, FACT_Purchase_Order,
                  FACT_Inventory_Snapshot, FACT_Vehicle_Trip,
                  FACT_Customer_Order, FACT_Return_Shipment, FACT_Claim
  Dim tables    : DIM_Customer, DIM_Carrier, DIM_Driver, DIM_Vehicle,
                  DIM_Warehouse, DIM_Route, DIM_Product, DIM_Supplier,
                  DIM_Geography, DIM_Date, DIM_Service_Type, DIM_Reason_Code

KEY METRICS AND EXACT SQL:
  Total shipments          : COUNT(Shipment_ID) from FACT_Shipment
  On-time delivery rate    : COUNT WHERE Actual_Delivery_Date <= Promised_Delivery_Date
                             / COUNT(Shipment_ID) * 100
  OTIF rate                : COUNT WHERE On_Time = True AND Qty_Delivered = Qty_Ordered
                             / COUNT(Shipment_ID) * 100
  Late deliveries          : COUNT WHERE Actual_Delivery_Date > Promised_Delivery_Date
  Average transit days     : AVG(DATEDIFF(day, Pickup_Date, Delivery_Date))
  Cost per delivery        : SUM(Freight_Cost_USD) / COUNT(Shipment_ID)
  Total freight cost       : SUM(Freight_Cost_USD) from FACT_Shipment
  Delivery success rate    : COUNT WHERE Delivery_Status = 'Delivered'
                             / COUNT(Shipment_ID) * 100
  Failed deliveries        : COUNT WHERE Delivery_Status IN ('Failed','Returned')
  Fleet utilization        : SUM(Load_Weight_Kg) / SUM(Max_Capacity_Kg) * 100
  Fuel cost                : SUM(Fuel_Cost_USD) from FACT_Vehicle_Trip
  Fuel efficiency          : SUM(Distance_Miles) / SUM(Fuel_Gallons_Used)
  Inventory accuracy       : Counted_Qty / System_Qty * 100
  Order fill rate          : SUM(Qty_Shipped) / SUM(Qty_Ordered) * 100
  Claims rate              : COUNT(Claim_ID) / COUNT(Shipment_ID) * 100
  Claim value              : SUM(Claim_Amount_USD) from FACT_Claim
  Dwell time hrs           : AVG time at facility from arrival to departure

BUSINESS VOCABULARY:
  OTD         : On-Time Delivery rate — primary SLA metric
  OTIF        : On-Time In-Full — on time AND correct quantity
  SLA         : Service Level Agreement — committed delivery timeframe
  POD         : Proof of Delivery — confirmation signature or code
  AWB         : Air Waybill — air freight tracking document
  BOL         : Bill of Lading — shipment document for ground or sea
  First mile  : Pickup from shipper to first sorting hub
  Last mile   : Final delivery leg to end customer — highest cost per unit
  Dwell time  : Time a shipment waits at a facility or hub
  LTL         : Less Than Truckload — partial load shipment
  FTL         : Full Truckload — dedicated truck
  Lead time   : Time from order placement to delivery
  Safety stock: Minimum inventory buffer to prevent stockout
  Reorder point: Inventory level that triggers a purchase order
  3PL         : Third-Party Logistics — outsourced logistics provider
  Carrier     : Transportation company — DIM_Carrier
  Lane        : Origin-destination route pair
  Reverse logistics: Return shipments — FACT_Return_Shipment

COMMON FILTERS:
  Completed deliveries  : Delivery_Status = 'Delivered'
  Exclude cancelled     : Shipment_Status != 'Cancelled'
  Active carriers       : Carrier_Status = 'Active'
  Outbound only         : Shipment_Direction = 'Outbound'
  Exclude test          : Is_Test = False

STANDARD FK RELATIONSHIPS:
  FACT_Shipment.Customer_ID         -> DIM_Customer.Customer_ID
  FACT_Shipment.Carrier_ID          -> DIM_Carrier.Carrier_ID
  FACT_Shipment.Origin_Warehouse_ID -> DIM_Warehouse.Warehouse_ID
  FACT_Shipment.Dest_Geography_ID   -> DIM_Geography.Geography_ID
  FACT_Shipment.Ship_Date_ID        -> DIM_Date.Date_ID
  FACT_Shipment.Service_Type_ID     -> DIM_Service_Type.Service_Type_ID
  FACT_Delivery_Event.Shipment_ID   -> FACT_Shipment.Shipment_ID
  FACT_Delivery_Event.Driver_ID     -> DIM_Driver.Driver_ID
  FACT_Vehicle_Trip.Vehicle_ID      -> DIM_Vehicle.Vehicle_ID
  FACT_Vehicle_Trip.Driver_ID       -> DIM_Driver.Driver_ID
  FACT_Vehicle_Trip.Route_ID        -> DIM_Route.Route_ID
  FACT_Warehouse_Movement.Product_ID-> DIM_Product.Product_ID
  FACT_Claim.Shipment_ID            -> FACT_Shipment.Shipment_ID
  FACT_Claim.Reason_Code_ID         -> DIM_Reason_Code.Reason_Code_ID
""",

# ══════════════════════════════════════════════════════════════════════════════
"Other": """
SCHEMA CONVENTION: Analyse tables as-is based on actual column names,
data types, and sample values. No domain preconceptions.

APPROACH:
  1. Identify which tables are FACT tables (contain numeric measures,
     many foreign keys, high row counts) versus DIM tables (descriptive
     attributes, fewer rows, primary keys referenced by fact tables).
  2. For each numeric column identify whether it is a measure to aggregate
     (SUM / AVG / COUNT) or a dimension to filter or group by.
  3. Identify FK relationships by matching ID column names across tables.
  4. Identify date columns for time-series analysis.
  5. Identify status and flag columns that act as common filters.
  6. Build business vocabulary based only on what the data actually shows.
     Do not invent columns or tables that are not present in the schema.
""",
}


def get_domain_context(domain: str) -> str:
    """Return the domain context string for a given domain name."""
    return DOMAIN_CONTEXT.get(domain, DOMAIN_CONTEXT["Other"])


def get_domain_list() -> list[str]:
    """Return the list of available domain names for the UI selector."""
    return DOMAINS
