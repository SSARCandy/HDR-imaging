:: A script that run all tone-mapping algorithm provide by tmo
:: Make sure your tm_*.exe is in you system PATH
:: ---
:: [Usage] tonemap.bat <filename without extension>

SET filename=%1

tm_ashikhmin.exe -i "%filename%.hdr" -o "%filename%-ashikhmin.jpg"
tm_gdc.exe -i "%filename%.hdr" -o "%filename%-gdc.jpg"
tm_miller.exe -i "%filename%.hdr" -o "%filename%-miller.jpg"
tm_retinex.exe -i "%filename%.hdr" -o "%filename%-retinex.jpg"
tm_bilateral.exe -i "%filename%.hdr" -o "%filename%-bilateral.jpg"
tm_histadj.exe -i "%filename%.hdr" -o "%filename%-histadj.jpg"
tm_mom.exe -i "%filename%.hdr" -o "%filename%-mom.jpg"
tm_schlick.exe -i "%filename%.hdr" -o "%filename%-schlick.jpg"
tm_chiu.exe -i "%filename%.hdr" -o "%filename%-chiu.jpg"
tm_horn.exe -i "%filename%.hdr" -o "%filename%-horn.jpg"
tm_oppenheim.exe -i "%filename%.hdr" -o "%filename%-oppenheim.jpg"
tm_tr.exe -i "%filename%.hdr" -o "%filename%-tr.jpg"
tm_drago.exe -i "%filename%.hdr" -o "%filename%-drago.jpg"
tm_icam.exe -i "%filename%.hdr" -o "%filename%-icam.jpg"
tm_pattanaik.exe -i "%filename%.hdr" -o "%filename%-pattanaik.jpg"
tm_trilateral.exe -i "%filename%.hdr" -o "%filename%-trilateral.jpg"
tm_ferschin.exe -i "%filename%.hdr" -o "%filename%-ferschin.jpg"
tm_linear.exe -i "%filename%.hdr" -o "%filename%-linear.jpg"
tm_photographic.exe -i "%filename%.hdr" -o "%filename%-photographic.jpg"
tm_ward.exe -i "%filename%.hdr" -o "%filename%-ward.jpg"
tm_ferwerda.exe -i "%filename%.hdr" -o "%filename%-ferwerda.jpg"
tm_log.exe -i "%filename%.hdr" -o "%filename%-log.jpg"
tm_photoreceptor.exe -i "%filename%.hdr" -o "%filename%-photoreceptor.jpg"
tm_yee.exe -i "%filename%.hdr" -o "%filename%-yee.jpg"