% BTU Computer Engineering Undergraduate Thesis LaTeX Template
% Aligned with: BTU_BM_Tez_Yazım_Sablonu_2026_updated.docx
% Engine: XeLaTeX / Tectonic compatible.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{btu-thesis}[2026/06/11 Bursa Technical University BM thesis template]

\LoadClass[12pt,a4paper,oneside]{report}

% --- Page geometry ---
\RequirePackage[a4paper,left=4cm,right=2.5cm,top=2.5cm,bottom=2.5cm,headsep=0pt,includehead=false,includefoot=false,footskip=1.5cm]{geometry}

% --- Font and language ---
\RequirePackage{fontspec}
% The DOCX template uses Times New Roman as its document default.
% If Microsoft fonts are unavailable, Tinos is the closest metrically compatible fallback.
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{%
  \IfFontExistsTF{Tinos}{\setmainfont{Tinos}}{\setmainfont{TeX Gyre Termes}}%
}
% The outer cover in the DOCX is Arial-like; prefer real Arial, then metrically compatible fallbacks.
\IfFontExistsTF{Arial}{\setsansfont{Arial}}{%
  \IfFontExistsTF{Arimo}{\setsansfont{Arimo}}{%
    \IfFontExistsTF{Liberation Sans}{\setsansfont{Liberation Sans}}{\setsansfont{TeX Gyre Heros}}%
  }%
}
\RequirePackage{polyglossia}
\setdefaultlanguage{turkish}
\RequirePackage{csquotes}
\RequirePackage{unicode-math}
% Times-compatible math font; keeps equations closer to the Word/Times requirement.
\IfFontExistsTF{XITSMath-Regular.otf}{\setmathfont{XITSMath-Regular.otf}}{%
  \IfFontExistsTF{STIXTwoMath-Regular.otf}{\setmathfont{STIXTwoMath-Regular.otf}}{}%
}

% --- Core packages ---
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{tikz}
\RequirePackage{setspace}
\RequirePackage{ragged2e}
\RequirePackage{titlesec}
\RequirePackage{tocloft}
\RequirePackage{caption}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{longtable}
\RequirePackage{pdflscape}
\RequirePackage{fancyhdr}
\RequirePackage{etoolbox}
\RequirePackage{amsmath}
\RequirePackage{hyperref}
\RequirePackage{bookmark}

% --- PDF metadata / link behavior ---
\hypersetup{hidelinks, unicode=true, pdfcreator={LaTeX with btu-thesis.cls}}

% --- Global paragraph/body style ---
\AtBeginDocument{%
  \renewcommand{\contentsname}{İÇİNDEKİLER}%
  \renewcommand{\listfigurename}{ŞEKİL LİSTESİ}%
  \renewcommand{\listtablename}{ÇİZELGE LİSTESİ}%
  \renewcommand{\figurename}{Şekil}%
  \renewcommand{\tablename}{Çizelge}%
  \fontsize{12pt}{18pt}\selectfont
  \onehalfspacing
  \justifying
  \setlength{\parindent}{0pt}% Word template body starts flush-left.
  \setlength{\parskip}{6pt}% Approximates 6 pt before/after Word body paragraphs.
}

% --- Page numbers: bottom centered, no header ---
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\fancyfoot[C]{\thepage}
\fancypagestyle{plain}{%
  \fancyhf{}%
  \renewcommand{\headrulewidth}{0pt}%
  \renewcommand{\footrulewidth}{0pt}%
  \fancyfoot[C]{\thepage}%
}
\fancypagestyle{btunofooter}{%
  \fancyhf{}%
  \renewcommand{\headrulewidth}{0pt}%
  \renewcommand{\footrulewidth}{0pt}%
}

% --- Section heading style ---
% First degree heading: bold, uppercase text supplied by the author, 72 pt before / 18 pt after.
\titleformat{\chapter}[hang]
  {\normalfont\bfseries\normalsize}
  {\thechapter.}{0.45em}{}
\titlespacing*{\chapter}{0pt}{72pt}{18pt}

\titleformat{\section}[hang]
  {\normalfont\bfseries\normalsize}
  {\thesection}{0.45em}{}
\titlespacing*{\section}{0pt}{18pt}{12pt}

\titleformat{\subsection}[hang]
  {\normalfont\bfseries\normalsize}
  {\thesubsection}{0.45em}{}
\titlespacing*{\subsection}{0pt}{12pt}{6pt}

\titleformat{\subsubsection}[hang]
  {\normalfont\bfseries\normalsize}
  {\thesubsubsection}{0.45em}{}
\titlespacing*{\subsubsection}{0pt}{12pt}{6pt}

\titleformat{\paragraph}[hang]
  {\normalfont\bfseries\normalsize}
  {}{0pt}{}
\titlespacing*{\paragraph}{0pt}{12pt}{6pt}

\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

% Ensure chapter starts on a new page. Use \cleardoublepage manually if your department requires right-hand openings.
\let\BTUoldchapter\chapter
\renewcommand{\chapter}{\clearpage\BTUoldchapter}

% --- Contents/list formatting ---
\renewcommand{\contentsname}{İÇİNDEKİLER}
\renewcommand{\listfigurename}{ŞEKİL LİSTESİ}
\renewcommand{\listtablename}{ÇİZELGE LİSTESİ}
\renewcommand{\figurename}{Şekil}
\renewcommand{\tablename}{Çizelge}

\renewcommand{\cftdotsep}{0.6}
\renewcommand{\cftchapfont}{\bfseries}
\renewcommand{\cftchappagefont}{\bfseries}
\renewcommand{\cftchapleader}{\bfseries\cftdotfill{\cftdotsep}}
\renewcommand{\cftsecfont}{\normalfont}
\renewcommand{\cftsecpagefont}{\normalfont}
\renewcommand{\cftsubsecfont}{\normalfont}
\renewcommand{\cftsubsecpagefont}{\normalfont}
\renewcommand{\cftsubsubsecfont}{\normalfont}
\renewcommand{\cftsubsubsecpagefont}{\normalfont}
\setlength{\cftbeforechapskip}{0pt}
\setlength{\cftbeforesecskip}{0pt}
\setlength{\cftbeforesubsecskip}{0pt}
\setlength{\cftbeforesubsubsecskip}{0pt}
\setlength{\cftchapindent}{0pt}
\setlength{\cftsecindent}{20pt}
\setlength{\cftsubsecindent}{40pt}
\setlength{\cftsubsubsecindent}{60pt}
\renewcommand{\cftchapaftersnum}{.}
\setlength{\cftchapnumwidth}{28pt}
\setlength{\cftsecnumwidth}{34pt}
\setlength{\cftsubsecnumwidth}{46pt}
\setlength{\cftsubsubsecnumwidth}{58pt}
\renewcommand{\cftfigpresnum}{\bfseries Şekil~}
\renewcommand{\cftfigaftersnum}{ : \normalfont}
\setlength{\cftfigindent}{0pt}
\setlength{\cftfignumwidth}{59pt}
\renewcommand{\cfttabpresnum}{\bfseries Çizelge~}
\renewcommand{\cfttabaftersnum}{ : \normalfont}
\setlength{\cfttabindent}{0pt}
\setlength{\cfttabnumwidth}{78pt}
% Keep TOC/list headings aligned with the graduate LaTeX template's 72 pt
% first-level heading rhythm instead of tocloft's near-top default.
\setlength{\cftbeforetoctitleskip}{53pt}
\setlength{\cftaftertoctitleskip}{0pt}
\setlength{\cftbeforeloftitleskip}{53pt}
\setlength{\cftafterloftitleskip}{0pt}
\setlength{\cftbeforelottitleskip}{53pt}
\setlength{\cftafterlottitleskip}{0pt}
\renewcommand{\cfttoctitlefont}{\bfseries\normalsize}
\renewcommand{\cftloftitlefont}{\bfseries\normalsize}
\renewcommand{\cftlottitlefont}{\bfseries\normalsize}
\renewcommand{\cfttoctitlefont}{\bfseries\normalsize}
\renewcommand{\cftaftertoctitle}{\hfill\mbox{}\par\vspace{18pt}\hfill\underline{Sayfa}\par\vspace{6pt}}
\renewcommand{\cftafterloftitle}{\hfill\mbox{}\par\vspace{18pt}\hfill\underline{Sayfa}\par\vspace{6pt}}
\renewcommand{\cftafterlottitle}{\hfill\mbox{}\par\vspace{18pt}\hfill\underline{Sayfa}\par\vspace{6pt}}

% Keep list entries single-spaced.
\pretocmd{\tableofcontents}{\begingroup\singlespacing\sloppy\setlength{\emergencystretch}{2em}\setlength{\parskip}{0pt}}{}{}
\apptocmd{\tableofcontents}{\endgroup}{}{}
\pretocmd{\listoffigures}{\begingroup\singlespacing\sloppy\setlength{\emergencystretch}{2em}\setlength{\parskip}{0pt}}{}{}
\apptocmd{\listoffigures}{\endgroup}{}{}
\pretocmd{\listoftables}{\begingroup\singlespacing\sloppy\setlength{\emergencystretch}{2em}\setlength{\parskip}{0pt}}{}{}
\apptocmd{\listoftables}{\endgroup}{}{}

% --- Captions ---
\DeclareCaptionLabelSeparator{btucolon}{ : }
\DeclareCaptionFont{btucaption}{\fontsize{10pt}{12pt}\selectfont}
\captionsetup{font=btucaption, labelfont=bf, labelsep=btucolon, justification=justified, singlelinecheck=true}
\captionsetup[figure]{position=bottom, skip=6pt, name=Şekil}
\captionsetup[table]{position=top, skip=6pt, name=Çizelge}

% --- Turkish-style chapter numbering for figures/tables/equations ---
\numberwithin{equation}{chapter}

% --- Lists of abbreviations/symbols helpers ---
\newenvironment{BTUabbr}[1]{%
  \BTUfrontheading{#1}\addcontentsline{toc}{chapter}{#1}%
  \singlespacing\setlength{\parskip}{0pt}%
  \begin{tabular}{@{}>{\bfseries}p{3.2cm}@{ {\bfseries :} }p{10.8cm}@{}}%
}{\end{tabular}\onehalfspacing}

% --- Title metadata defaults. Override in metadata.tex. ---
\newcommand{\BTUUniversity}{BURSA TEKNİK ÜNİVERSİTESİ}
\newcommand{\BTUFaculty}{MÜHENDİSLİK VE DOĞA BİLİMLERİ FAKÜLTESİ}
\newcommand{\BTUDepartment}{Bilgisayar Mühendisliği Bölümü}
\newcommand{\BTUThesisType}{LİSANS BİTİRME ÇALIŞMASI}
\newcommand{\BTUThesisTitleTR}{BİTİRME ÇALIŞMASI BAŞLIĞI BURAYA GELİR\\GEREKLİ İSE İKİNCİ SATIR\\GEREKLİ İSE ÜÇÜNCÜ SATIR, ÜÇ SATIRA SIĞDIRINIZ}
\newcommand{\BTUThesisTitleEN}{THESIS TITLE IN ENGLISH HERE}
\newcommand{\BTUThesisTitleApproval}{BİTİRME ÇALIŞMASI BAŞLIĞI}
\newcommand{\BTUStudentName}{Öğrenci Adı SOYADI}
\newcommand{\BTUStudentNumber}{Öğrenci No}
\newcommand{\BTUStudentCoverBlock}{\BTUStudentName}
\newcommand{\BTUStudentInnerCoverBlock}{\BTUStudentName\\(\BTUStudentNumber)}
\newcommand{\BTUPlagiarismSignatureBlock}{%
  \begin{minipage}{9.5cm}
  \centering
  \begin{tabular}{@{}r@{}l@{}}
  Öğrencinin Adı Soyadı: & \hspace{0.2cm}\BTUStudentName\\[0.95cm]
  İmzası: & \\%
  \end{tabular}%
  \end{minipage}%
}
\newcommand{\BTUAdvisor}{Ünvan Ad SOYAD}
\newcommand{\BTUAdvisorFull}{Prof. Dr. Adı SOYADI}
\newcommand{\BTUJuryOne}{Dr. Öğr. Üyesi  Adı SOYADI}
\newcommand{\BTUJuryTwo}{Öğr. Gör. Dr.  Adı SOYADI}
\newcommand{\BTUChair}{Doç. Dr. Adı Soyadı}
\newcommand{\BTUSecondAdvisor}{}
\newcommand{\BTUSubmissionMonthYear}{TEZİN SAVUNULDUĞU AY YIL}
\newcommand{\BTUDefenseDate}{28 Haziran 2017}
\newcommand{\BTUForewordDate}{Haziran 2017}
\newcommand{\BTUKeywordsTR}{anahtar kelime 1, anahtar kelime 2, anahtar kelime 3}
\newcommand{\BTUKeywordsEN}{keyword 1, keyword 2, keyword 3}
\newcommand{\BTULogoPath}{figures/btu-logo.jpeg}
\newcommand{\BTUCoverBackgroundPath}{figures/btu-cover-background.png}
\definecolor{BTUCoverBlue}{HTML}{0B4E73}

% Outer cover uses equal side margins in the DOCX, so it is centered on the page.
\newcommand{\BTUOuterCoverX}{10.50cm}
% Inner cover/front matter text block center is centered in the Word text area:
% left 4 cm + text width/2 = 11.25 cm.
\newcommand{\BTUCoverX}{11.25cm}

\newcommand{\BTUcovernode}[3][]{%
  \node[align=center, text width=14.5cm, #1] at ([xshift=\BTUCoverX,yshift=-#2]current page.north west) {#3};%
}

\newcommand{\BTUcovertextnode}[3][]{%
  \node[anchor=north, inner sep=0pt, align=center, text width=14.5cm, #1] at ([xshift=\BTUCoverX,yshift=-#2]current page.north west) {#3};%
}

\newcommand{\BTUoutercovertextnode}[3][]{%
  \node[anchor=north, inner sep=0pt, align=center, text width=14.5cm, #1] at ([xshift=\BTUOuterCoverX,yshift=-#2]current page.north west) {#3};%
}

\newcommand{\BTUmakecover}{%
  \clearpage\thispagestyle{btunofooter}%
  {\singlespacing
  \begin{tikzpicture}[remember picture, overlay]
    \node[inner sep=0pt] at (current page.center) {\includegraphics[width=\paperwidth,height=\paperheight]{\BTUCoverBackgroundPath}};
    \BTUoutercovertextnode[text=BTUCoverBlue, font=\normalfont\bfseries\fontsize{16pt}{18pt}\selectfont]{6.36cm}{T.C.}
    \BTUoutercovertextnode[text=BTUCoverBlue, font=\normalfont\bfseries\fontsize{16pt}{18pt}\selectfont]{7.01cm}{\BTUUniversity}
    \BTUoutercovertextnode[text=BTUCoverBlue, font=\normalfont\bfseries\fontsize{15pt}{17pt}\selectfont]{7.65cm}{\BTUFaculty}
    \BTUoutercovertextnode[text=BTUCoverBlue, font=\normalfont\bfseries\fontsize{15pt}{17pt}\selectfont]{10.70cm}{BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ}
    \BTUoutercovertextnode[text=BTUCoverBlue, font=\normalfont\bfseries\fontsize{12pt}{13.8pt}\selectfont]{14.83cm}{\BTUThesisTitleTR}
    \BTUoutercovertextnode[text=BTUCoverBlue, font=\normalfont\bfseries\fontsize{14pt}{16pt}\selectfont]{20.74cm}{\BTUThesisType}
    \BTUoutercovertextnode[text=BTUCoverBlue, font=\normalfont\fontsize{12pt}{14pt}\selectfont]{21.81cm}{\BTUStudentCoverBlock}
    \BTUoutercovertextnode[text=BTUCoverBlue, font=\normalfont\bfseries\fontsize{12pt}{14pt}\selectfont]{26.67cm}{\BTUSubmissionMonthYear}
  \end{tikzpicture}%
  }%
  \null\clearpage
}

\newcommand{\BTUmakeinnercover}{%
  \thispagestyle{btunofooter}%
  {\singlespacing
  \begin{tikzpicture}[remember picture, overlay]
    \BTUcovertextnode[font=\normalfont\bfseries\fontsize{12pt}{13.8pt}\selectfont]{4.01cm}{T.C.}
    \BTUcovertextnode[font=\normalfont\bfseries\fontsize{12pt}{13.8pt}\selectfont]{4.49cm}{\BTUUniversity}
    \BTUcovertextnode[font=\normalfont\bfseries\fontsize{12pt}{13.8pt}\selectfont]{4.98cm}{\BTUFaculty}
    \BTUcovertextnode[font=\normalfont\bfseries\fontsize{12pt}{13.8pt}\selectfont]{7.33cm}{BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ}
    \BTUcovertextnode[font=\normalfont\bfseries\fontsize{12pt}{13.8pt}\selectfont]{11.36cm}{\BTUThesisTitleTR}
    \BTUcovertextnode[font=\normalfont\bfseries\fontsize{12pt}{13.8pt}\selectfont]{17.34cm}{\BTUThesisType}
    \BTUcovertextnode[font=\normalfont\bfseries\fontsize{12pt}{13.8pt}\selectfont]{18.20cm}{\BTUStudentInnerCoverBlock}
    \BTUcovertextnode[font=\normalfont\bfseries\fontsize{12pt}{13.8pt}\selectfont]{22.17cm}{Danışman: \BTUAdvisor}
    \ifdefempty{\BTUSecondAdvisor}{}{%
      \BTUcovertextnode[font=\normalfont\bfseries\fontsize{12pt}{13.8pt}\selectfont]{23.57cm}{İkinci Danışman: \BTUSecondAdvisor}%
    }%
    \BTUcovertextnode[font=\normalfont\bfseries\fontsize{12pt}{13.8pt}\selectfont]{25.87cm}{\BTUSubmissionMonthYear}
  \end{tikzpicture}%
  }%
  \null\clearpage
}


\newcommand{\BTUmakeapprovalpage}{%
  \clearpage\thispagestyle{btunofooter}%
  {\singlespacing\setlength{\parskip}{0pt}\fontsize{12pt}{14pt}\selectfont
  \begin{tikzpicture}[remember picture, overlay]
    \node[anchor=north west, text width=14.5cm, align=justify] at ([xshift=4.0cm,yshift=-5.00cm]current page.north west) {BTÜ, Mühendislik ve Doğa Bilimleri Fakültesi Bilgisayar Mühendisliği Bölümü'nün \BTUStudentNumber{} numaralı öğrencileri \BTUStudentName, ilgili yönetmeliklerin belirlediği gerekli tüm şartları yerine getirdikten sonra hazırladıkları “\BTUThesisTitleApproval” başlıklı bitirme çalışmasını aşağıda imzaları olan jüri önünde başarı ile sunmuşlardır.};

    \node[anchor=north west] at ([xshift=4.0cm,yshift=-8.60cm]current page.north west) {\textbf{Danışmanı :}};
    \node[anchor=north west] at ([xshift=7.85cm,yshift=-8.60cm]current page.north west) {\textbf{\BTUAdvisorFull}};
    \node[anchor=north west] at ([xshift=15.55cm,yshift=-8.86cm]current page.north west) {..............................};
    \node[anchor=north west] at ([xshift=7.85cm,yshift=-9.08cm]current page.north west) {Bursa Teknik Üniversitesi};

    \node[anchor=north west] at ([xshift=4.0cm,yshift=-12.55cm]current page.north west) {\textbf{Jüri Üyeleri :}};
    \node[anchor=north west] at ([xshift=7.85cm,yshift=-12.55cm]current page.north west) {\textbf{\BTUJuryOne}};
    \node[anchor=north west] at ([xshift=15.55cm,yshift=-12.85cm]current page.north west) {..............................};
    \node[anchor=north west] at ([xshift=7.85cm,yshift=-13.05cm]current page.north west) {Bursa Teknik Üniversitesi};

    \node[anchor=north west] at ([xshift=7.85cm,yshift=-14.55cm]current page.north west) {\textbf{\BTUJuryTwo}};
    \node[anchor=north west] at ([xshift=15.55cm,yshift=-14.85cm]current page.north west) {..............................};
    \node[anchor=north west] at ([xshift=7.85cm,yshift=-15.05cm]current page.north west) {Bursa Teknik Üniversitesi};

    \node[anchor=north west] at ([xshift=4.0cm,yshift=-22.35cm]current page.north west) {\textbf{Savunma Tarihi :}};
    \node[anchor=north west] at ([xshift=7.85cm,yshift=-22.35cm]current page.north west) {\BTUDefenseDate};

    \node[anchor=north west] at ([xshift=4.0cm,yshift=-23.95cm]current page.north west) {\textbf{BM Bölüm Başkanı : \BTUChair}};
    \node[anchor=north west] at ([xshift=6.55cm,yshift=-24.65cm]current page.north west) {..............................};
    \node[anchor=north west] at ([xshift=7.85cm,yshift=-25.00cm]current page.north west) {Bursa Teknik Üniversitesi};
    \node[anchor=north west] at ([xshift=15.55cm,yshift=-25.00cm]current page.north west) {....../......../.......};
  \end{tikzpicture}%
  }
  \null\clearpage
}


\newcommand{\BTUmakeplagiarism}{%
  \clearpage\thispagestyle{btunofooter}%
  {\singlespacing\setlength{\parskip}{0pt}\fontsize{12pt}{14pt}\selectfont
  \begin{tikzpicture}[remember picture, overlay]
    \node[anchor=north, font=\bfseries] at ([xshift=\BTUCoverX,yshift=-4.83cm]current page.north west) {İNTİHAL BEYANI};
    \node[anchor=north west, text width=14.5cm, align=justify] at ([xshift=4.0cm,yshift=-5.84cm]current page.north west) {Bu bitirme çalışmasında görsel, işitsel ve yazılı biçimde sunulan tüm bilgi ve sonuçların akademik ve etik kurallara uyularak tarafımızdan elde edildiğini, bitirme çalışması içinde yer alan ancak bu çalışmaya özgü olmayan tüm sonuç ve bilgileri bitirme çalışmasında kaynak göstererek belgelediğimizi, aksinin ortaya çıkması durumunda her türlü yasal sonucu kabul ettiğimizi beyan ederiz.};
    \node[anchor=north, align=center, text width=14.5cm] at ([xshift=\BTUCoverX,yshift=-8.70cm]current page.north west) {\BTUPlagiarismSignatureBlock};
  \end{tikzpicture}%
  }
  \null\clearpage
}


\newcommand{\BTUfrontheading}[1]{%
  \clearpage
  \vspace*{1.65cm}%
  {\bfseries #1\par}%
  \vspace{0.72cm}%
}

\newcommand{\BTUforewordheading}{%
  \clearpage
  \vspace*{1.02cm}%
  {\bfseries ÖNSÖZ\par}%
  \vspace{0.13cm}%
}

\newcommand{\BTUfrontmatter}{%
  \clearpage
  \pagenumbering{roman}
  \setcounter{page}{1}
}
\newcommand{\BTUmainmatter}{%
  \clearpage
  \pagenumbering{arabic}
  \setcounter{page}{1}
  \onehalfspacing
}

\newenvironment{BTUforeword}{%
  \BTUforewordheading\addcontentsline{toc}{chapter}{ÖNSÖZ}%
  \singlespacing\setlength{\parskip}{6pt}%
}{\onehalfspacing}

\newcommand{\BTUforewordsignature}{%
  \vspace{1.1cm}\noindent\BTUForewordDate\hfill\BTUStudentName\par
}

\newenvironment{BTUabstractTR}{%
  \clearpage\singlespacing\setlength{\parskip}{6pt}%
  \begin{center}\bfseries\BTUThesisTitleTR\end{center}%
  \vspace{18pt}%
  \begin{center}\bfseries ÖZET\end{center}%
  \addcontentsline{toc}{chapter}{ÖZET}%
}{%
  \par\vspace{1\baselineskip}\noindent\textbf{Anahtar kelimeler:} \BTUKeywordsTR\par\onehalfspacing
}

\newenvironment{BTUabstractEN}{%
  \clearpage\singlespacing\setlength{\parskip}{6pt}%
  \begin{center}\bfseries\BTUThesisTitleEN\end{center}%
  \vspace{18pt}%
  \begin{center}\bfseries SUMMARY\end{center}%
  \addcontentsline{toc}{chapter}{SUMMARY}%
}{%
  \par\vspace{1\baselineskip}\noindent\textbf{Keywords:} \BTUKeywordsEN\par\onehalfspacing
}

\newcommand{\BTUaddcontents}{\clearpage\def\contentsname{İÇİNDEKİLER}\addcontentsline{toc}{chapter}{İÇİNDEKİLER}\tableofcontents}
\newcommand{\BTUaddtablelist}{\clearpage\def\listtablename{ÇİZELGE LİSTESİ}\def\tablename{Çizelge}\addcontentsline{toc}{chapter}{ÇİZELGE LİSTESİ}\listoftables}
\newcommand{\BTUaddfigurelist}{\clearpage\def\listfigurename{ŞEKİL LİSTESİ}\def\figurename{Şekil}\addcontentsline{toc}{chapter}{ŞEKİL LİSTESİ}\listoffigures}

\newlength{\BTUrefhang}
\setlength{\BTUrefhang}{2.5cm}
\newenvironment{BTUreferences}{%
  \BTUfrontheading{KAYNAKLAR}\addcontentsline{toc}{chapter}{KAYNAKLAR}%
  \singlespacing\setlength{\parskip}{0pt}\justifying
  \begin{list}{}{%
    \setlength{\leftmargin}{\BTUrefhang}%
    \setlength{\itemindent}{-\BTUrefhang}%
    \setlength{\labelwidth}{0pt}%
    \setlength{\labelsep}{0pt}%
    \setlength{\itemsep}{6pt}%
    \setlength{\parsep}{0pt}%
    \setlength{\topsep}{0pt}%
  }%
}{\end{list}\onehalfspacing}

\newcommand{\BTUappendices}{%
  \clearpage\appendix
  \chapter*{EKLER}\addcontentsline{toc}{chapter}{EKLER}%
}

\newenvironment{BTUcv}{%
  \BTUfrontheading{ÖZGEÇMİŞ}\addcontentsline{toc}{chapter}{ÖZGEÇMİŞ}%
  \singlespacing\setlength{\parskip}{6pt}%
}{\onehalfspacing}
