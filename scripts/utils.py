"""
    Helper utility functions dictionaries
"""
import sys

REGEXES = {
    'IMSS'      :"(institutos?.{0,2})?m[ée]xicano.{0,2}del.{0,2}seguro.{0,2}social|imss",
    'SEMARNAT'  :"secretar[ií][oa]s?.{0,2}del.{0,2}medio.{0,2}ambiente.{0,2}y.{0,2}recursos?.{0,2}naturales|semarnat",
    'SHCP'      :"secretar[íi][oa]s?.{0,5}de.{0,2}hacienda|hacienda.{0,2}y.{0,2}crédito.{0,2}p[úu]blico|shcp",
    'CFE'       :"(comisi[oó]n.{0,2})?federal.{0,2}de.{0,2}electricidad|cfe",
    'CONAGUA'   :"(comisi[óo]n.{0,2})?nacional.{0,2}del.{0,2}agua|conagua",
    'SEDENA'    :"secretar[íi][oa]s?.{0,2}de.{0,2}la.{0,2}defensa|de.{0,2}la.{0,2}defensa.{0,2}nacional|sede[nñ]a",
    'PEMEX'     :"p[ée]tr[oó]l[ée]os?.{0,2}m[eé]xicanos?|p[ée]mex",
    'SEDESOL'   :"secretar[íi][oa]s?.{0,5}de.{0,2}desarrollo.{0,2}social|sedesol",
    'COFEPRIS'  :"(comisi[óo]n.{0,2})?federal.{0,2}para.{0,2}la.{0,2}protecci[óo]n contra.{0,2}riesgos?.{0,2}sanitarios?|cofepris",
    'SAGARPA'   :"secretar[ií][oa]s?.{0,2}de.{0,2}agricultura|de.{0,2}agricultura.{0,2}ganader[ií]a.{0,2}desarrollo|sagarpa",
    'SCT'       :"secretar[íi][oa]s?.{0,2}de.{0,2}comunicaciones.{0,2}y.{0,2}transportes?|sct",
    'SFP'       :"secretar[íi][ao]s?.{0,2}de.{0,2}la.{0,2}funci[óo]n.{0,2}p[uú]blica|sfp",
    'PGR'       :"procurad[ou]r([ií]?a)?.{0,2}general.{0,2}de.{0,2}la.{0,2}rep[uú]blica|pgr",
    'SEGOB'     :"secretar[íi][oa]s?.{0,5}de.{0,2}gobernac[ií][oó]n|segob",
    'SRE'       :"secretar[íi][oa]s?.{0,5}de.{0,2}relaciones?.{0,2}exteriores?|sre",
    'IMPI'      :"(instituto.{0,2})?mexicano.{0,2}de.{0,2}la.{0,2}propiedad.{0,2}industrial|impi",
    'SSP'       :"secretar[ií][oa]s?.{0,2}de.{0,2}seguridad.{0,2}p[uú]blica|ssp",
    'INM'       :"(instituto.{0,2})?nacional.{0,2}de.{0,2}migraci[oó]n|inm",
    'SENER'     :"secretar[íi][oa]s?.{0,2}de.{0,2}energ[ií]a|sener",
    'SEP'       :"secretar[íi][oa]s?.{0,5}de.{0,2}educaci[oó]n.{0,2}p[uú]blica",
    'SSA'       :"secretar[íi][oa]s?.{0,2}de.{0,5}salud|ssa",
    'SEECO'     :"secretar[ií][oa]s?.{0,2}de.{0,2}econom[ií]a|seeco"  
}


# Progress bar. Ignore.
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))