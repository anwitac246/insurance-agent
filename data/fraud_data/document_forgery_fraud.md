# Fraud Case Studies - Document Forgery

## Case 201: The Forged Police FIR
* **Fraud Type**: Forged Official Documents
* **Scenario**: Claimant submitted an FIR for a stolen vehicle. The vehicle was allegedly stolen from a mall parking lot.
* **Detection Indicators**:
  - The FIR number format did not match the standard sequential format used by the local precinct.
  - The signature of the Station House Officer (SHO) was a low-resolution copy-paste from a public document.
  - Verification call to the local precinct confirmed no such FIR was registered.
* **Resolution**: Claim denied. Reported to authorities for forgery of official police documents.

## Case 202: The Inflated Repair Estimate
* **Fraud Type**: Altered Repair Bills
* **Scenario**: Claimant submitted a repair estimate for $8,500 from an out-of-network garage for a minor rear-end collision.
* **Detection Indicators**:
  - The PDF invoice had multiple text layers; the total amount field had a different font (Arial) compared to the rest of the document (Helvetica).
  - OCR extraction flagged inconsistent alignment in the "Parts Replaced" column.
  - The cost of a replacement bumper was listed at 400% above the manufacturer's MSRP.
* **Resolution**: Claim partially approved. Payment adjusted to the standard market rate of $1,200 based on a network garage reassessment.

## Case 203: The Fake Driver's License
* **Fraud Type**: Identity / License Forgery
* **Scenario**: The vehicle was driven by the policyholder's 19-year-old son, who was excluded from the policy. To claim coverage, the policyholder submitted a forged license claiming a friend (25 years old) was driving.
* **Detection Indicators**:
  - The license number failed the API checksum validation with the Department of Motor Vehicles.
  - The micro-print pattern on the scanned license was missing (a common sign of a cheap digital forgery).
* **Resolution**: Claim rejected due to misrepresentation of the driver.

## Case 204: The Fabricated Medical Bill
* **Fraud Type**: Medical Billing Fraud
* **Scenario**: Following a minor collision with zero visible vehicle damage, the claimant submitted a $15,000 hospital bill for severe whiplash and spinal surgery.
* **Detection Indicators**:
  - The letterhead matched a real hospital, but the medical billing codes (CPT codes) listed were for gastrointestinal surgery, not orthopedics.
  - The dates of "surgery" coincided with dates the claimant was actively posting vacation photos on social media.
* **Resolution**: Bodily injury claim denied. Investigated for medical fraud.

## Case 205: Ghost Garage Receipts
* **Fraud Type**: Phantom Business Invoices
* **Scenario**: Claimant submitted a final repair bill of $4,000 from "Elite Auto Body" to claim a cash reimbursement.
* **Detection Indicators**:
  - A query to the state business registry revealed "Elite Auto Body" had its business license revoked 5 years prior.
  - The address listed on the invoice currently belongs to a residential apartment complex.
* **Resolution**: Claim denied. 

## Case 206: The Altered Policy Cover Note
* **Fraud Type**: Policy Inception Fraud
* **Scenario**: Claimant submits a PDF of their policy cover note showing coverage started on May 1st. The accident occurred on May 2nd.
* **Detection Indicators**:
  - The internal system database shows the policy was actually purchased on May 3rd.
  - PDF metadata analysis shows the document was edited using Adobe Acrobat on May 4th to change the "03" to a "01".
* **Resolution**: Claim outright rejected. Policy cancelled.
