import * as React from 'react';
import { IconButton, Stack, Button } from '@mui/material';
import './style.css'
import { useState } from 'react'
import Info from './Info'
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
import original from '../images/original.png'
import changed from '../images/changed.png'
import clock_grey from '../icons/clock_grey.png'
import calendar_grey from '../icons/calendar_grey.png'
import eraser from '../icons/eraser.png'
import eraser_grey from '../icons/eraser_grey.png'
import { useNavigate } from "react-router-dom";
import image1 from "../images/image1.png"
import image2 from "../images/image2.png"


export default function UnlearningResult(){
    const navigate = useNavigate();

    const [popup, setPopup] = useState(false)
    const [idx, setIdx] = useState(0)
    const InfoPopupHandler = () => {
        setPopup(true)
    }
    const btnHandler = (idx) => {
        setIdx(idx)
    }
    const header_font_style = {fontSize: 25, marginTop: 0, marginBottom: 0, marginLeft: 5, fontFamily: "Roboto", color: "white"}
    const header_bg_style_short = { p: 1, bgcolor: "#505050", width: 450, textAlign: "left"}
    const header_bg_style_long = { p: 1, bgcolor: "#505050", width: 1070, color: "white", textAlign: "left"}
    const subheader_font_style = {fontSize: 16, marginTop: 0, marginBottom: 0, marginLeft: 5, fontFamily: "Roboto", color: "#6D6D6D"}
    const subheader_bg_style = { p: 1, bgcolor: "#D9D9D9", width: 335, height: 20, textAlign: "left", marginTop: 2}
    const content_bg_style_entire = { p: 1, bgcolor: "#F5F5F5", width: 450, height: 530, textAlign: "left", marginTop: 2}
    const content_font_style_my = {fontFamily: "Roboto", color: "#6D6D6D", fontSize: 25, marginTop: 3}
    const content_bg_style_my = { p: 1, bgcolor: "#F5F5F5", width: 335, height: 45, textAlign: "left"}
    const img_size = {width: 350, height: 300}


    const tempdata = [
        {"req_num": 1, "unlearning_method": "Fine Tuning", "unlearning_time": "3 min 50 sec", "conducted_date": "2023. 11. 29. 15:10", "mia_before": 1.876, "mia_after": 2.228, "zero": 1.2},
        {"req_num": 2, "unlearning_method": "Fine Tuning", "unlearning_time": "2 min 45 sec", "conducted_date": "2023. 11. 29. 15:45", "mia_before": 1.452, "mia_after": 2.345, "zero": 1.2},
        {"req_num": 3, "unlearning_method": "Fine Tuning", "unlearning_time": "3 min 14 sec", "conducted_date": "2023. 11. 29. 15:50", "mia_before": 1.823, "mia_after": 2.231, "zero": 1.6},
        {"req_num": 4, "unlearning_method": "Fine Tuning", "unlearning_time": "1 min 12 sec", "conducted_date": "2023. 11. 29. 16:10", "mia_before": 1.563, "mia_after": 2.657, "zero": 1.7},
        {"req_num": 5, "unlearning_method": "Fine Tuning", "unlearning_time": "2 min 10 sec", "conducted_date": "2023. 11. 29. 16:20", "mia_before": 1.674, "mia_after": 2.394, "zero": 1.8},
        {"req_num": 6, "unlearning_method": "Fine Tuning", "unlearning_time": "2 min 10 sec", "conducted_date": "2023. 11. 29. 17:13", "mia_before": 1.234, "mia_after": 2.293, "zero": 1.9},
        {"req_num": 7, "unlearning_method": "Fine Tuning", "unlearning_time": "2 min 10 sec", "conducted_date": "2023. 11. 29. 18:30", "mia_before": 1.112, "mia_after": 2.058, "zero": 1.8}

    ]

    return(
        <div>
            <Stack direction="row">
                <IconButton aria-label="delete" 
                            sx = {{marginRight: 30, marginTop: -3, color: 'black', justifyContent: 'flex-end'}}
                            onClick = {() => InfoPopupHandler()}>
                    <img  src={eraser} style = {{marginLeft: 30, marginTop: 30}}/>
                </IconButton>
            </Stack>
            <div>
                <Stack direction = "row">
                    <h1 style  = {{fontSize: 60, fontFamily: "Roboto", color: "#247AFC",  textAlign: "left", marginLeft: 120, marginTop: -10, marginRight: 860}}>Result Dashboard</h1>
                    <Button variant="contained" sx = {{width: 200, height: 80, borderRadius: "50px", 
                            fontSize: "1.5rem", fontFamily: "Roboto", textTransform: 'none', fontWeight: "200px"}} 
                            onClick = {() => navigate("/Home")}>
                        <h3> Return </h3>
                    </Button>
                </Stack>
                <Stack direction = "row">
                    <Stack direction = "column"  sx = {{marginLeft: 15}}>
                        <Box component="section" sx={header_bg_style_short}>
                            <h3 style = {header_font_style}>
                                Entire Requests
                            </h3>
                        </Box>
                        <Box component="section" sx={content_bg_style_entire}>
                            {
                                tempdata.map((v, i) => {
                                    
                                    return (
                                        <Stack direction = "column">
                                            {
                                            (idx+1 === tempdata[i].req_num)
                                            ?
                                            <Button variant="contained" sx = {{marginBottom: 2}} onClick = {() => btnHandler(i)}>
                                                Trial #{tempdata[i].req_num} {tempdata[i].conducted_date}
                                            </Button>
                                            :
                                            <Button variant="outlined" sx = {{marginBottom: 2}} onClick = {() => btnHandler(i)}>
                                                Trial #{tempdata[i].req_num} {tempdata[i].conducted_date}
                                            </Button>
                                            }
                                        </Stack>
                                        
                                    )
                                })
                            }
                            
                        </Box>
                    </Stack>
                    <Stack direction = "column" sx = {{marginLeft: 3}}>
                        <Stack direction = "row">
                            <Box component="section" sx={header_bg_style_long}>
                                <h3 style = {header_font_style}>
                                    My Request
                                </h3>
                            </Box>
                        </Stack>
                        <Stack direction = "row">
                            <Stack direction = "column">
                                <Box component="section" sx={subheader_bg_style}>
                                    <h3 style = {subheader_font_style}>
                                        Unlearning Method
                                    </h3>
                                </Box>
                                <Box component="section" sx={content_bg_style_my}>
                                    <Stack direction = "row">
                                        <img src = {eraser_grey} style = {{width: 50, height: 40, marginLeft: 5, marginTop: 5, marginRight: 10}}/>
                                        <h3 style = {content_font_style_my}>
                                            {tempdata[idx].unlearning_method}
                                        </h3>
                                    </Stack>
                                </Box>
                            </Stack>

                            <Stack direction = "column" sx = {{marginLeft: 2}}>
                                <Box component="section" sx={subheader_bg_style}>
                                    <h3 style = {subheader_font_style}>
                                        Unlearning Time
                                    </h3>
                                </Box>
                                <Box component="section" sx={content_bg_style_my}>
                                    <Stack direction = "row">
                                        <img src = {clock_grey} style = {{width: 33, height: 37, marginLeft: 5, marginTop: 5, marginRight: 13}}/>
                                        <h3 style = {content_font_style_my}>
                                            {tempdata[idx].unlearning_time}
                                        </h3>
                                    </Stack>
                                    
                                </Box>
                            </Stack>

                            <Stack direction = "column" sx = {{marginLeft: 2}}>
                                <Box component="section" sx={subheader_bg_style}>
                                    <h3 style = {subheader_font_style}>
                                        Conducted Date
                                    </h3>
                                </Box>
                                <Box component="section" sx={content_bg_style_my}>
                                    <Stack direction = "row">
                                        <img src = {calendar_grey} style = {{width: 33, height: 34, marginLeft: 5, marginTop: 5, marginRight: 13}}/>
                                        <h3 style = {content_font_style_my}>
                                            {tempdata[idx].conducted_date}
                                        </h3>
                                    </Stack>
                                </Box>
                            </Stack>
                        </Stack>
                        <Stack direction = "row" sx = {{marginTop: 2}}>
                            <Stack direction = "column">
                                <img src = {original} style ={img_size}></img>
                                <p style = {{marginTop: 3}}>*Original image</p>
                            </Stack>
                            <Stack direction = "column" sx = {{marginLeft: 2}}>
                                <img src = {image1} style ={img_size}></img>
                                <p style = {{marginTop: 3}}>*Before unlearning: Your image inside the AI</p>
                            </Stack>
                            <Stack direction = "column" sx = {{marginLeft: 2}}>
                                <img src = {image2} style ={img_size}></img>
                                <p style = {{marginTop: 3}}>*After unlearning: Your image inside the AI</p>
                            </Stack>
                        </Stack>
                        <Stack direction = "row" sx = {{marginTop: -2}}>
                            <Stack direction = "column">
                                <Box component="section" sx={subheader_bg_style}>
                                    <h3 style = {subheader_font_style}>
                                        MIA (Before Unlearning)
                                    </h3>
                                </Box>
                                <Box component="section" sx={content_bg_style_my}>
                                    <h3 style = {content_font_style_my}>
                                        {tempdata[idx].mia_before}
                                    </h3>
                                </Box>
                            </Stack>

                            <Stack direction = "column" sx = {{marginLeft: 2}}>
                                <Box component="section" sx={subheader_bg_style}>
                                    <h3 style = {subheader_font_style}>
                                        MIA (After Unlearning)
                                    </h3>
                                </Box>
                                <Box component="section" sx={content_bg_style_my}>
                                    <h3 style = {content_font_style_my}>
                                        {tempdata[idx].mia_after}
                                    </h3>
                                </Box>
                            </Stack>
                            <Stack direction = "column" sx = {{marginLeft: 2}}>
                                <Box component="section" sx={subheader_bg_style}>
                                    <h3 style = {subheader_font_style}>
                                        Zero Retrain Forgetting Score
                                    </h3>
                                </Box>
                                <Box component="section" sx={content_bg_style_my}>
                                    <h3 style = {content_font_style_my}>
                                        {tempdata[idx].zero}
                                    </h3>
                                </Box>
                            </Stack>

                        </Stack>
                    </Stack>
                </Stack>
            </div>
            {(popup) && <Info setPopup = {setPopup}/>}
        </div>
    )
    
}
