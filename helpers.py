def get_seed_urls_test() ->list[str]:
    '''
    Returns short list of subset urls.
    '''
    urls = [
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/W4-11.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/G21EA.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/G21IP.html"
    ]
    return urls

def get_seed_urls() -> list[str]:
    '''
    Returns a list of all subset urls.
    '''
    urls = [
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/W4-11.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/G21EA.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/G21IP.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/DIPCS10.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/PA26.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/SIE4x4.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/ALKBDE10.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/YBDE18.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/AL2X6.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/HEAVYSB11.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/NBPRC.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/ALK8.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/RC21.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/G2RC.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/BH76RC.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/FH51.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/TAUT15.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/DC13.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/MB16-43.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/DARC.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/RSE43.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/BSR36.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/CDIE20.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/ISO34.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/ISOL24.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/C60ISO.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/PArel.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/BH76.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/BHPERI.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/BHDIV10.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/INV24.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/BHROT27.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/PX13.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/WCPT18.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/RG18.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/ADIM6.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/S22.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/S66.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/HEAVY28.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/WATER27.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/CARBHB12.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/PNICO23.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/HAL59.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/AHB21.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/CHB6.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/IL16.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/IDISP.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/ICONF.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/ACONF.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/Amino20x4.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/PCONF21.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/MCONF.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/SCONF.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/UPU23.html",
    "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/BUT14DIOL.html"
]

    return urls