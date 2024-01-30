from fpdf import FPDF
from PIL import Image
import json
import os
import pandas as pd
import io
from datetime import datetime
from ReportGenerationUtils import *

# Set metadata path
metadata_path = os.path.join("../SampleCase/MRI/session_info.xlsx")

# Read metadata into DataFrame
df = pd.read_excel(metadata_path, converters={'study_id': str}, index_col='session_id', sheet_name='info')

# Get unique patient IDs
session_id = sorted(os.listdir('../SampleCase/images'))
ans = [y for x, y in df.groupby('study_id', as_index=False)]

# Read age data
df_age = pd.read_excel(metadata_path, converters={'study_id': str}, index_col='study_id', sheet_name='age')

# Set path for saving reports
path_new = '../SampleCase/reports/'
isExist = os.path.exists(path_new)
if not isExist:
    os.makedirs(path_new)

# Loop through each patient's sessions
for ix in ans:
    session_ids = ix['time_point']
    patient_id = str(ix['study_id'].values[0])

    # Initialize PDF for summary report
    pdf_summary = init_summary_pdf(ix, df_age)

    # Initialize features dictionary
    dict_features = {'session': [], 'date': [], 'PostOp': [], 'numofSlices': [], 'maxExtraMeatalDiameter': [],
                     'maxAxialDiameter': [], 'max3dDiameter': [], 'VolumeIntra': [], 'VolumeExtra': [],
                     'VolumeWhole': [],
                     'maxExtraMeatalDiameterChange': [], 'maxAxialDiameterChange': [], 'max3dDiameterChange': [],
                     'VolumeIntraChange': [], 'VolumeExtraChange': [], 'VolumeWholeChange': []}

    # Set print status and summary index
    print_status = False
    idx_summary_y = 25
    idx = 0

    # Loop through each session
    for sess_ in session_ids:
        sess_id_path = '../SampleCase/features_json/%s_%s_features.json' % (patient_id, sess_)
        image_path = '../SampleCase/images/'

        # Check if features file exists
        if os.path.isfile(sess_id_path):
            print_status = True

            # Read features from JSON file
            with open(sess_id_path, 'r') as openfile:
                json_object = json.load(openfile)

            # Populate features dictionary
            id_name = json_object['ID']
            time_point = json_object['Time_point']
            dict_features['session'].append(sess_)
            dict_features['date'].append(datetime.strptime(sess_[3:], '%Y%m%d').date())
            dict_features['PostOp'].append(ix[ix['time_point'] == sess_]['Postop'].values[0])
            dict_features['numofSlices'].append(json_object['Positive_planes'])
            dict_features = GetFeatureDict(json_object, dict_features)

            # Open and crop MRI image
            img = Image.open(image_path + "%s_%s_MRI.png" % (patient_id, sess_))
            col_start, col_end = crop_image_only_outside(img, tol=0)
            img_cropped = img.crop((0, col_start, img.size[0], col_end))
            ratio = img_cropped.size[1] / img_cropped.size[0]

            # Add image to PDF
            pdf_summary.image(img_cropped, x=10, y=idx_summary_y + 3, w=80, h=80 * ratio)
            pdf_summary.set_xy(10, idx_summary_y)
            pdf_summary.set_text_color(r=0, g=0, b=0)
            pdf_summary.write(text="Time Point %s  t" % (idx + 1))
            pdf_summary.char_vpos = "SUB"
            pdf_summary.write(text="%s" % (idx + 1))
            pdf_summary.char_vpos = "LINE"
            pdf_summary.write(text="  Date: %s " % (datetime.strptime(sess_[3:], '%Y%m%d').date()))

            # Add "Post-Operation" label if applicable
            if ix[ix['time_point'] == sess_]['Postop'].values[0] == 1:
                pdf_summary.write(text="  Post-Operation")

            # Add linear measurements information
            pdf_summary.set_xy(100, idx_summary_y + 4)
            if dict_features['maxExtraMeatalDiameter'][idx] == 0.0 or ix[ix['time_point'] == sess_]['Postop'].values[
                0] == True:
                linear_feature, text = 'maxAxialDiameter', 'Axial'
            else:
                linear_feature, text = 'maxExtraMeatalDiameter', 'Extrameatal'

            if dict_features['numofSlices'][idx] > 1:
                pdf_summary.multi_cell(120, 4,
                                       text="Maximum %s Diameter: %.1f mm  \nWhole Tumor Volume: %.1f mm\u00b3 \nExtrameatal Volume: %.1f mm\u00b3 "
                                            % (text, round(dict_features[linear_feature][idx], 1),
                                               round(dict_features['VolumeWhole'][idx], 1),
                                               round(dict_features['VolumeExtra'][idx], 1)), align='L', markdown=True)
            else:
                pdf_summary.multi_cell(120, 4, text="Maximum %s Diameter: %.1f mm" % (
                text, round(dict_features[linear_feature][idx], 1)), align='L', markdown=True)

            idx_summary_y += (80 * ratio + 4)
            idx += 1

    # Determine linear measurement and text based on conditions
    if (dict_features['maxExtraMeatalDiameterChange'].count(0) == len(
            dict_features['maxExtraMeatalDiameterChange'])) or (any(dict_features['PostOp']) == True):
        linear_measurement, text = 'maxAxialDiameter', 'Axial'
    else:
        linear_measurement, text = 'maxExtraMeatalDiameter', 'EM'

    # Add guide and diagrams to PDF
    if len(dict_features['session']) == 1:
        add_guide(pdf_summary)
    else:
        pdf_summary = add_diagram(pdf_summary, dict_features, linear_measurement)
        pdf_summary = plot_graph(dict_features, pdf_summary, linear_measurement, ix, text)
        pdf_summary = add_guide(pdf_summary, linear_measurement)

    # Save the summary report
    if print_status == True:
        pdf_summary.output(path_new + "/summaryreport_%s.pdf" % (patient_id))
