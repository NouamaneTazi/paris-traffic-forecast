amont_aval = [
    ["Av_Champs_Elysees-Washington", "Av_Georges_V-Place_Dunant"],
    ["Pl_Concorde-Av_Champs_Elysees", "Av_Champs_Elysees-Dutuit"],
    ["Av_Champs_Elysees-Colisee", "Rond_Point_Champs_Elysees"],
    ["Av_Champs_Elysees-La_Boetie", "Av_Champs_Elysees-Berri"],
    ["Concorde_Ouest", "Pl_Concorde-Av_Champs_Elysees"],
    ["Av_Champs_Elysees-Berri", "Av_Champs_Elysees-Washington"],
    ["Av_Champs_Elysees-Face_Air_Franc", "Av_Champs_Elysees-Balzac"],
    ["Pl_Concorde-Av_Champs_Elysees", "Cours_la_Reine-Concorde"],
    ["Champs-Tilsitt", "Av_Champs_Elysees-Face_Air_Franc"],
    ["Cours_la_Reine-Concorde", "Sortie_Souterrain_Champs_Elysees"],
    ["Sortie_Souterrain_Champs_Elysees", "Tuileries-Sedar_Senghor"],
    ["Av_Champs_Elysees-Face_Air_Franc", "Grande_Armee-Forge"],
    ["Av_Champs_Elysees-Clemenceau", "Rond_Point_Champs_Elysees"],
    ["Rond_Point_Champs_Elysees", "Av_Champs_Elysees-Clemenceau"],
    ["Av_Champs_Elysees-Washington", "Av_Champs_Elysees-Berri"],
    ["Av_Champs_Elysees-Colisee", "Av_Champs_Elysees-La_Boetie"],
    ["Av_Champs_Elysees-Dutuit", "Av_Champs_Elysees-Clemenceau"],
    ["Rond_Point_Champs_Elysees", "Av_Champs_Elysees-Colisee"],
    ["Av_Champs_Elysees-Balzac", "Av_Champs_Elysees-Washington"],
    ["Av_Champs_Elysees-Berri", "Av_Champs_Elysees-La_Boetie"],
    ["Av_Georges_V-Place_Dunant", "Av_Champs_Elysees-Washington"],
]

amont_aval += [
    ["Convention-St_Charles", "Convention-Lourmel"],
    ["Convention-Felix_Faure", "Convention-Nivert"],
    ["Convention-Lourmel", "Convention-St_Charles"],
    ["Convention-Blomet", "Convention-Vaugirard"],
    ["Convention-St_Charles", "Convention-Gutemberg"],
    ["Convention-Felix_Faure", "Convention-Lourmel"],
    ["Convention-Lourmel", "Convention-Felix_Faure"],
    ["Convention-Nivert", "Convention-Felix_Faure"],
    ["Faure-Javel", "Convention-Felix_Faure"],
    ["Convention-Gutemberg", "Convention-St_Charles"],
    ["Convention-Nivert", "Lecourbe-Convention"],
    ["Convention-Gutemberg", "Rond_Point_Mirabeau"],
    ["Convention-Vaugirard", "Convention-Olivier_de_Serres"],
    ["Convention-Felix_Faure", "Faure-Javel"],
    ["Lecourbe-Convention", "Convention-Nivert"],
    ["Rond_Point_Mirabeau", "Convention-Gutemberg"],
    ["Lecourbe-Convention", "Convention-Blomet"],
    ["Convention-Olivier_de_Serres", "Place_Charles_Valin"],
    ["Lecourbe-Convention", "Lecourbe-Croix-Nivert"],
]

amont_aval += [
    ["Sevres-Babylone", "Sevres-Sts_Peres"],
    ["Sts_Peres-Voltaire", "Sts_Peres-Universite"],
    ["Bd_St_Germain-St_Guillaume", "Bd_St_Germain-Sts_Peres"],
    ["Sts_Peres-Grenelle", "Sevres-Sts_Peres"],
    ["Sts_Peres-Universite", "Bd_St_Germain-Sts_Peres"],
    ["Malaquais-Bonaparte", "Sts_Peres-Voltaire"],
    ["Bd_St_Germain-Sts_Peres", "Sts_Peres-Grenelle"],
    ["Bd_St_Germain-Sts_Peres", "Bd_St_Germain-Dragon"],
]

# overwrite links.txt to save the new links
with open("data_src/links.txt", "w") as f:
    for amont, aval in amont_aval:
        f.write(
            f"https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/download/?format=csv&sort=t_1h&facet=libelle&facet=t_1h&facet=libelle_nd_amont&facet=libelle_nd_aval&facet=etat_barre&refine.t_1h=2021&refine.libelle_nd_amont={amont}&refine.libelle_nd_aval={aval}&use_labels_for_header=false&csv_separator=%3B\n"
        )
        f.write(f"    out={amont}__{aval}.csv\n")
