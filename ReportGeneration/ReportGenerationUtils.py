from fpdf import FPDF
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np


def crop_image_only_outside(im, tol=0):
    gray = im.convert('L')
    img = np.asarray(gray)
    mask = img > tol
    m, n = np.shape(img)
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()

    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return row_start, row_end


def init_pdf(ix):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('helvetica', size=14, style="B", )
    pdf.write(text="Sample Report\n\n")
    patient_id = str(ix['study_id'].values[0])
    pdf.set_font('helvetica', size=12)
    pdf.write(text=("Patient ID:  " + patient_id + "\n"))
    return pdf


def init_summary_pdf(ix, df_age):
    patient_id = str(ix['study_id'].values[0])
    patient_age = str(df_age[df_age.index == patient_id]['Age'].values[0])
    patient_gender = str(df_age[df_age.index == patient_id]['Gender'].values[0])
    symptoms = str(df_age[df_age.index == patient_id]['ClinicalInfo'].values[0])
    additional_info = str(ix['AdditionalInfo'].values[0])
    pdf_summary = FPDF()
    pdf_summary.add_page()
    pdf_summary.set_font('helvetica', size=12, style="B", )
    pdf_summary.write(text="Summary Report\n")
    patient_id = str(ix['study_id'].values[0])
    pdf_summary.set_font('helvetica', size=9)
    pdf_summary.write(text=("Patient ID: %s \n" % (patient_id)))
    pdf_summary.write(text=("Age: %s \t Gender: %s \n" % (patient_age, patient_gender)))
    pdf_summary.write(text=("Symptoms: %s " % (symptoms)))
    if not additional_info == np.nan:
        pdf_summary.write(text=("(%s)\n" % (additional_info)))
    return pdf_summary


def GetFeature(json_object, dict_features, feature):
    time_dim_feature = json_object[feature]
    if feature not in dict_features.keys():
        dict_features[feature] = time_dim_feature
    else:
        dict_features[feature].append(time_dim_feature)
    return dict_features


def percentage_change(init_dim,next_dim):
    if init_dim == 0:
        return 0
    else:
        return round(((next_dim - init_dim)/init_dim)*100,3)

def GetFeatureChange(json_object, dict_features, feature):
    time_dim_feature = dict_features[feature][-1]
    Volume_features = ['VolumeIntra', 'VolumeExtra', 'VolumeWhole']
    Linear_features = ["maxExtraMeatalDiameter", "maxAxialDiameter", "max3dDiameter"]
    if len(dict_features["%sChange" % (feature)]) == 0:
        dict_features["%sChange" % (feature)].append(0)
    elif len(dict_features["%sChange" % (feature)]) > 0:
        if feature in Volume_features:
            dict_features["%sChange" % (feature)].append(
                percentage_change(dict_features[feature][-2], time_dim_feature))
        else:
            dict_features["%sChange" % (feature)].append(time_dim_feature - dict_features[feature][-2])
    return dict_features


def GetFeatureDict(json_object, dict_features):
    if (json_object["VolumeIntra"] > 0 and json_object["VolumeExtra"] > 0) or (
            json_object["VolumeIntra"] == 0 and json_object["VolumeExtra"] > 0):
        if not json_object["maxExtraMeatalDiameter"] == None:
            feature_list = ["maxExtraMeatalDiameter", "maxAxialDiameter", "max3dDiameter", "VolumeExtra", "VolumeIntra",
                            "VolumeWhole"]
            for feature in feature_list:
                dict_features = GetFeature(json_object, dict_features, feature)
                dict_features = GetFeatureChange(json_object, dict_features, feature)
        else:
            print('yes')
            dict_features["maxExtraMeatalDiameter"].append(0)
            dict_features = GetFeatureChange(json_object, dict_features, feature="maxExtraMeatalDiameter")
            feature_list = ["maxAxialDiameter", "max3dDiameter", "VolumeExtra", "VolumeIntra", "VolumeWhole"]
            for feature in feature_list:
                dict_features = GetFeature(json_object, dict_features, feature)
                dict_features = GetFeatureChange(json_object, dict_features, feature)
    elif json_object["VolumeIntra"] > 0 and json_object["VolumeExtra"] == 0:
        dict_features["maxExtraMeatalDiameter"].append(0)
        dict_features = GetFeatureChange(json_object, dict_features, feature="maxExtraMeatalDiameter")
        feature_list = ["maxAxialDiameter", "max3dDiameter", "VolumeExtra", "VolumeIntra", "VolumeWhole"]
        for feature in feature_list:
            dict_features = GetFeature(json_object, dict_features, feature)
            dict_features = GetFeatureChange(json_object, dict_features, feature)
    else:
        feature_list = ["maxExtraMeatalDiameter", "maxAxialDiameter", "max3dDiameter", "VolumeExtra", "VolumeIntra",
                        "VolumeWhole"]
        for feature in feature_list:
            dict_features[feature].append(0)
            dict_features = GetFeatureChange(json_object, dict_features, feature)
    return dict_features


def GetAdditionalFeatureDict(json_object, dict_features):
    if (json_object["VolumeIntra"] > 0 and json_object["VolumeExtra"] > 0) or (
            json_object["VolumeIntra"] == 0 and json_object["VolumeExtra"] > 0):
        if not (json_object["maxIAMParallelDiameter"] == None or json_object["maxEMParallelDiameter"] == None):
            if json_object["maxIAMParallelDiameter"] < json_object["maxEMParallelDiameter"]:
                feature_list = ["maxExtraMeatalDiameter", "maxEMParallelDiameter", "maxEMPerpendicularDiameter",
                                "maxAxialDiameter", "max3dDiameter", "VolumeExtra", "VolumeIntra", "VolumeWhole"]
                for feature in feature_list:
                    dict_features = GetFeature(json_object, dict_features, feature)
                    dict_features = GetFeatureChange(json_object, dict_features, feature)
            else:
                features_neglected = ["maxExtraMeatalDiameter", "maxEMParallelDiameter", "maxEMPerpendicularDiameter"]
                for feature in features_neglected:
                    dict_features[feature].append(0)
                    dict_features = GetFeatureChange(json_object, dict_features, feature)
                feature_list = ["maxAxialDiameter", "max3dDiameter", "VolumeExtra", "VolumeIntra", "VolumeWhole"]
                for feature in feature_list:
                    dict_features = GetFeature(json_object, dict_features, feature)
                    dict_features = GetFeatureChange(json_object, dict_features, feature)
        else:
            feature_list = ["maxExtraMeatalDiameter", "maxEMParallelDiameter", "maxEMPerpendicularDiameter",
                            "max3dDiameter", "VolumeExtra", "VolumeIntra", "VolumeWhole"]
            for feature in feature_list:
                dict_features = GetFeature(json_object, dict_features, feature)
                dict_features = GetFeatureChange(json_object, dict_features, feature)
    elif json_object["VolumeIntra"] > 0 and json_object["VolumeExtra"] == 0:
        features_neglected = ["maxExtraMeatalDiameter", "maxEMParallelDiameter", "maxEMPerpendicularDiameter"]
        for feature in features_neglected:
            dict_features[feature].append(0)
            dict_features = GetFeatureChange(json_object, dict_features, feature)
        feature_list = ["maxAxialDiameter", "max3dDiameter", "VolumeExtra", "VolumeIntra", "VolumeWhole"]
        for feature in feature_list:
            dict_features = GetFeature(json_object, dict_features, feature)
            dict_features = GetFeatureChange(json_object, dict_features, feature)
    else:
        feature_list = ["maxExtraMeatalDiameter", "maxEMParallelDiameter", "maxEMPerpendicularDiameter",
                        "maxAxialDiameter", "max3dDiameter", "VolumeExtra", "VolumeIntra", "VolumeWhole"]
        for feature in feature_list:
            dict_features[feature].append(0)
            dict_features = GetFeatureChange(json_object, dict_features, feature)
    return dict_features


def set_Color_vol(change_EMvol):
    if change_EMvol >= 20:
        r, g, b = 255, 0, 0
    elif 10 <= change_EMvol < 20:
        r, g, b = 255, 128, 0
    elif change_EMvol < 10:
        r, g, b = 0, 102, 0
    return r, g, b


def set_Color_diameter(change_maxEMPara):
    if change_maxEMPara >= 2:
        r, g, b = 255, 0, 0
    elif 0 <= change_maxEMPara < 2:
        r, g, b = 255, 128, 0
    elif change_maxEMPara < 0:
        r, g, b = 0, 102, 0
    return r, g, b


def add_diagram(pdf, dict_features, linear_measurement):
    x_im, y_im = 12, 165
    for idx in range(len(dict_features["session"])):
        if idx == 0:
            pdf.image("./icons/t%d.png" % (idx + 1), x=x_im, y=y_im, w=12, h=12)
        else:
            change_EMvol = dict_features["VolumeExtraChange"][idx]
            change_maxEM = dict_features["%sChange" % (linear_measurement)][idx]
            pdf.set_line_width(0.25)
            red, green, blue = set_Color_vol(change_EMvol)
            pdf.set_draw_color(r=red, g=green, b=blue)
            pdf.set_text_color(r=red, g=green, b=blue)
            pdf.line(x1=x_im + 12, y1=y_im + 5, x2=x_im + 30, y2=y_im + 5)
            pdf.set_xy(x_im + 14, y_im + 2)
            if dict_features["numofSlices"][idx - 1] <= 1 or dict_features["numofSlices"][idx] <= 1:
                pdf.multi_cell(0, 1, text="**N/A**", align='L', markdown=True)
            else:
                if change_EMvol == 0:
                    pdf.multi_cell(0, 1, text="**N/A**", align='L', markdown=True)
                else:
                    pdf.multi_cell(0, 1, text="**%s%%** "
                                              % (round(change_EMvol, 1)), align='L', markdown=True)

            red, green, blue = set_Color_diameter(change_maxEM)
            pdf.set_draw_color(r=red, g=green, b=blue)
            pdf.line(x1=x_im + 12, y1=y_im + 7, x2=x_im + 30, y2=y_im + 7)
            pdf.set_text_color(r=red, g=green, b=blue)
            pdf.set_xy(x_im + 14, y_im + 8.5)
            pdf.multi_cell(0, 1, text="**%s mm** "
                                      % (round(change_maxEM, 2)), align='L', markdown=True)

            x_im = x_im + 30
            pdf.image("./icons/t%d.png" % (idx + 1), x=x_im, y=y_im, w=12, h=12)

    return pdf


def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def axis_data_coords_sys_transform(axis_obj_in, xin, yin, inverse=False):
    xlim = axis_obj_in.get_xlim()
    ylim = axis_obj_in.get_ylim()

    xdelta = xlim[1] - xlim[0]
    ydelta = ylim[1] - ylim[0]

    xout = xlim[0] + xin * xdelta
    yout = ylim[0] + yin * ydelta

    return xout, yout

def plot_graph(dict_features, pdf_summary, linear_measurement, ix, text):
    mpl.style.use('bmh')
    plt.rcParams['font.family'] = 'Sans-Serif'
    plt.rcParams['font.size'] = '14'
    mpl.rcParams['figure.dpi'] = 300
    labels = [0]
    for xx in range(0, len(dict_features['date']) - 1):
        labels.append(diff_month(dict_features['date'][xx + 1], dict_features['date'][xx]) + labels[-1])

    list_diff = [x - dict_features[linear_measurement][i - 1] for i, x in enumerate(dict_features[linear_measurement])][
                1:]

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])
    fig = plt.figure(figsize=(25, 6))
    ax = plt.subplot(gs[0, 0])
    plt.xlim([-1, max(labels) + 1])
    decision = list(ix['Decision'].values)

    # get the indices of with the slices more than 2
    label_idx = [i for i in range(len(dict_features['numofSlices'])) if dict_features['numofSlices'][i] > 1]
    intra_list = [dict_features['VolumeIntra'][idx] for idx in label_idx]
    extra_list = [dict_features['VolumeExtra'][idx] for idx in label_idx]
    labels_slices = [labels[idx] for idx in label_idx]
    ax.bar(labels_slices, intra_list, label='IntraMeatal Volume', color='#000066', edgecolor="black",
           width=max(labels) / 30)
    ax.bar(labels_slices, extra_list, bottom=intra_list, label='ExtraMeatal Volume', color='#006600', edgecolor="black",
           width=max(labels) / 30)

    ax.set_ylabel('Volume (mm$^3$)')
    ax.xaxis.set_label_coords(.5, -.3)
    lgd = ax.legend(bbox_to_anchor=(0.8, -1.0), ncol=2)

    ax2 = plt.subplot(gs[1, 0], sharex=ax)

    if len(extra_list) == 0 and len(intra_list) == 0:
        fig.delaxes(ax)
        for ix_tp in range(len(labels) - 1):
            decision_ = decision[ix_tp]
            ix_decision = labels[ix_tp]
            arr_image = Image.open('./icons/%s.png' % (decision_))
            arr_image = arr_image.resize((120, 120), Image.Resampling.LANCZOS)
            x0, y0 = ax2.transData.transform((ix_decision, 0))
            y0 += 150
            ax2.figure.figimage(arr_image, x0 + 100, y0, alpha=1.0)
    else:
        for ix_tp in range(len(labels) - 1):
            decision_ = decision[ix_tp]
            ix_decision = labels[ix_tp]
            arr_image = Image.open('./icons/%s.png' % (decision_))
            arr_image = arr_image.resize((140, 140), Image.Resampling.LANCZOS)
            x0, y0 = ax.transData.transform((ix_decision, 0))
            y0 += 150
            ax.figure.figimage(arr_image, x0 - 500, y0, alpha=1.0)

    if np.mean(list_diff) > 1:
        clip_on = True
    else:
        clip_on = False
        ax2.set_ylim(ymin=0, ymax=max(dict_features[linear_measurement]) + 5)
    ax2.set_ylim([min(dict_features[linear_measurement]) - 1, max(dict_features[linear_measurement]) + 1])
    ax2.hlines(y=dict_features[linear_measurement], xmin=-1, xmax=labels, linestyle="dashed", color='#DC143C',
               linewidth=1.0)
    ax2.plot(labels, dict_features[linear_measurement], clip_on=clip_on, linestyle='-', marker="X", markersize=12,
             linewidth=2.0, color='#DC143C', label='Max %s Diameter' % (text))
    ax2.set_xlabel('Time (months)')
    ax2.set_ylabel("Max %s diameter \n  (mm)" % (text))
    lgd2 = ax2.legend(bbox_to_anchor=(1.0, -0.45))
    ax2.yaxis.get_ticklocs(minor=True)
    ax2.minorticks_on()
    if ((round(max(dict_features[linear_measurement]))) - (round(min(dict_features[linear_measurement])) - 1)) < 5:
        ax2.set_yticks(np.arange(round(min(dict_features[linear_measurement])) - 1,
                                 round(max(dict_features[linear_measurement])) + 1, 1.0))
    elif ((round(max(dict_features[linear_measurement]))) - (round(min(dict_features[linear_measurement])) - 1)) < 12:
        ax2.set_yticks(np.arange(round(min(dict_features[linear_measurement])) - 1,
                                 round(max(dict_features[linear_measurement])) + 1, 3.0))
    else:
        ax2.set_yticks(np.arange(round(min(dict_features[linear_measurement])) - 1,
                                 round(max(dict_features[linear_measurement])) + 1, 6.0))

    # Turn off x-axis minor ticks
    ax2.xaxis.set_tick_params(which='minor', bottom=False)
    ax2.tick_params(axis='y', colors='#DC143C')
    for ax, color in zip([ax, ax2], ['black', 'black']):
        plt.setp(ax.spines.values(), color=color)
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=color)
    plt.subplots_adjust(hspace=.025)
    plt.savefig('output.png', bbox_extra_artists=(lgd, lgd2,), bbox_inches='tight', dpi=fig.dpi)
    im = Image.open('output.png')

    ratio = im.size[1] / im.size[0]
    pdf_summary.image(im, x=10, y=180, w=190, h=180 * ratio)
    return pdf_summary


def add_guide(pdf_summary, linear_measurement):
    pdf_summary.set_xy(11, 240)
    pdf_summary.set_text_color(r=0, g=0, b=0)
    pdf_summary.multi_cell(0, 1, text="**Comments**", align='L', markdown=True)
    pdf_summary.set_draw_color(r=0, g=0, b=0)
    pdf_summary.set_dash_pattern(dash=1, gap=1)
    pdf_summary.line(x1=10, y1=238, x2=200, y2=238)
    pdf_summary.line(x1=130, y1=238, x2=130, y2=260)
    pdf_summary.set_xy(132, 240)
    pdf_summary.set_text_color(r=0, g=0, b=0)
    pdf_summary.multi_cell(0, 1, text="**Decision**", align='L', markdown=True)

    pdf_summary.set_xy(11, 262)
    pdf_summary.set_text_color(r=0, g=0, b=0)
    pdf_summary.multi_cell(0, 1, text="**Guide**", align='L', markdown=True)
    pdf_summary.set_draw_color(r=0, g=0, b=0)
    pdf_summary.set_dash_pattern(dash=1, gap=1)
    pdf_summary.line(x1=10, y1=260, x2=200, y2=260)

    if linear_measurement == 'maxAxialDiameter':
        guide = Image.open('./icons/drawingguide_axial.png')
    else:
        guide = Image.open('./icons/drawingguide_em.png')
    ratio = guide.size[1] / guide.size[0]
    pdf_summary.image(guide, x=11, y=265, w=40, h=40 * ratio)

    index = Image.open('./icons/decisionindexv3.png')
    ratio = index.size[1] / index.size[0]
    pdf_summary.image(index, x=60, y=265, w=25, h=25 * ratio)

    pdf_summary.set_font('helvetica', size=8)
    pdf_summary.set_xy(95, 265)
    pdf_summary.multi_cell(0, 4,
                           text="**N/A** - Not Available (Tumour only visible on a single slice) \nIf both intra and extra regions are present, intrameatal region is always shown in **yellow**, Otherwise the **whole tumour is shown in green.**",
                           align='L', markdown=True)

    return pdf_summary