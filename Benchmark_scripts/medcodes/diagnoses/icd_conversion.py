from medcodes.diagnoses._mappers.icd9to10_dict import icd9to10dict
from medcodes.diagnoses._mappers.icd10to9_dict import icd10to9dict

def convert_9to10(code):
    if code in icd9to10dict.keys():
        return icd9to10dict[code]
    else:
        return code

def convert_10to9(code, digit3=False):
    if code in icd10to9dict.keys():
        output = icd10to9dict[code]
    else:
        output = code
    if digit3:
        output = output[:3]
    return output

def convert_9to10_list(icds, versions=None):
    out = []
    if versions==None:
        for code in icds:
            out.append(convert_9to10(code))
    else:
        for i, code in enumerate(icds):
            if versions[i]==9:
                out.append(convert_9to10(code))
            else:
                out.append(code)
    return out

def convert_10to9_list(icds, versions=None, digit3 = False):
    out = []
    if versions==None:
        for code in icds:
            out.append(convert_10to9(code, digit3))
    else:
        for i, code in enumerate(icds):
            if versions[i]==10:
                out.append(convert_10to9(code, digit3))
            else:
                out.append(code)
    return out