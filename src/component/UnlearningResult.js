import * as React from 'react';
import { IconButton, Stack, Button } from '@mui/material';
import './style.css'
import { useState, useEffect } from 'react'
import Info from './Info'
import Help from './Help'
import HelpZRF from './HelpZRF'
import HelpAttack from './HelpAttack'
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
import original from '../images/original.png'
import changed from '../images/changed.png'
import clock_grey from '../icons/clock_grey.png'
import calendar_grey from '../icons/calendar_grey.png'
import eraser from '../icons/eraser.png'
import eraser_grey from '../icons/eraser_grey.png'
import { useNavigate } from "react-router-dom";
import stage3 from '../icons/stage3.png'
import * as d3 from 'd3';
import result from '../result.csv'
import Papa from 'papaparse';
import HelpIcon from '@mui/icons-material/Help';


import image from '../uploads/image.png'
import original_model from '../uploads/original_model.png'
import unlearned_model from '../uploads/unlearned_model.png'
import difference from '../uploads/difference.png'

export default function UnlearningResult(){
    const navigate = useNavigate();
    const URLsplit = window.document.URL.split('/');
    const date = URLsplit[URLsplit.length - 1];
    const time = URLsplit[URLsplit.length - 2];
    const unlearnMIA = URLsplit[URLsplit.length - 3];
    const MIA = URLsplit[URLsplit.length - 4];
    const fileName = URLsplit[URLsplit.length - 5];

    const [popup, setPopup] = useState(false)
    const [helpPopup, setHelpPopup] = useState(false)
    const [helpPopupZRF, setHelpPopupZRF] = useState(false)
    const [helpPopupAttack, setHelpPopupAttack] = useState(false)
    const [idx, setIdx] = useState(0)
    const [data, setData] = useState([]);
    const [state, setState] = useState(false)

    const InfoPopupHandler = () => {
        setPopup(true)
    }
    const HelpPopupHandler = () => {
        setHelpPopup(true)
    }

    const HelpPopupZRFHandler = () => {
        setHelpPopupZRF(true)
    }
    const HelpPopupAttackHandler = () => {
        setHelpPopupAttack(true)
    }
    const btnHandler = (idx) => {
        setIdx(idx)
    }

    const header_font_style = {fontSize: 25, marginTop: 0, marginBottom: 0, marginLeft: 5, fontFamily: "Roboto", color: "white"}
    const header_bg_style_short = { p: 1, bgcolor: "#505050", width: 450, textAlign: "left"}
    const header_bg_style_long = { p: 1, bgcolor: "#505050", width: 1070, color: "white", textAlign: "left"}
    const subheader_font_style = {fontSize: 16, marginTop: 0, marginBottom: 0, marginLeft: 5, fontFamily: "Roboto", color: "#6D6D6D"}
    const subheader_bg_style = { p: 1, bgcolor: "#D9D9D9", width: 335, height: 20, textAlign: "left", marginTop: 2}
    const subheader_bg_style_wide = { p: 1, bgcolor: "#D9D9D9", width: 1070, height: 20, textAlign: "left", marginTop: 2}
    const content_bg_style_entire = { p: 1, bgcolor: "#F5F5F5", width: 450, height: 515, textAlign: "left", marginTop: 2}
    const content_font_style_my = {fontFamily: "Roboto", color: "#6D6D6D", fontSize: 25, marginTop: 3}
    const content_bg_style_my = { p: 1, bgcolor: "#F5F5F5", width: 335, height: 45, textAlign: "left"}
    const img_size = {width: 260, height: 230}


    const tempdata = [
        {"req_num": 1, "unlearning_method": "Negative Gradient", "unlearning_time": "3 min 50 sec", "conducted_date": "2023. 11. 29. 15:10", "mia_before": 1.876, "mia_after": 2.228, "zero": 1.2},
        {"req_num": 2, "unlearning_method": "Negative Gradient", "unlearning_time": "2 min 45 sec", "conducted_date": "2023. 11. 29. 15:45", "mia_before": 1.452, "mia_after": 2.345, "zero": 1.2},
        {"req_num": 3, "unlearning_method": "Negative Gradient", "unlearning_time": "3 min 14 sec", "conducted_date": "2023. 11. 29. 15:50", "mia_before": 1.823, "mia_after": 2.231, "zero": 1.6},
        {"req_num": 4, "unlearning_method": "Negative Gradient", "unlearning_time": "1 min 12 sec", "conducted_date": "2023. 11. 29. 16:10", "mia_before": 1.563, "mia_after": 2.657, "zero": 1.7},
        {"req_num": 5, "unlearning_method": "Negative Gradient", "unlearning_time": "2 min 10 sec", "conducted_date": "2023. 11. 29. 16:20", "mia_before": 1.674, "mia_after": 2.394, "zero": 1.8},
        {"req_num": 6, "unlearning_method": "Negative Gradient", "unlearning_time": "2 min 10 sec", "conducted_date": "2023. 11. 29. 17:13", "mia_before": 1.234, "mia_after": 2.293, "zero": 1.9},
        {"req_num": 7, "unlearning_method": "Negative Gradient", "unlearning_time": "2 min 10 sec", "conducted_date": "2023. 11. 29. 18:30", "mia_before": 1.112, "mia_after": 2.058, "zero": 1.8}
    ]


    return(
        <div>
            <Stack direction="row">
                <IconButton aria-label="delete" 
                            sx = {{marginRight: 30, marginTop: -3, color: 'black', justifyContent: 'flex-end'}}
                            onClick = {() => InfoPopupHandler()}>
                    <img  src={eraser} style = {{marginLeft: 30, marginTop: 30}}/>
                </IconButton>
                <Button variant="contained" sx = {{width: 300, height: 50, borderRadius: "50px", 
                            fontSize: "1rem", fontFamily: "Roboto", textTransform: 'none', fontWeight: "200px", marginTop: 4, marginLeft: 150}} 
                            onClick = {() => navigate("/Home")}>
                        <h3> Check Other Data </h3>
                </Button>
            </Stack>
            <div>
                <Stack direction = "row" justifyContent='center'>
                    {/* <h1 style  = {{fontSize: 60, fontFamily: "Roboto", color: "#247AFC",  textAlign: "left", marginLeft: 120, marginTop: -10, marginRight: 860}}>Result Dashboard</h1> */}
                    <img src = {stage3}></img>
                    {/* <Button variant="contained" sx = {{width: 400, height: 80, borderRadius: "50px", 
                            fontSize: "1.5rem", fontFamily: "Roboto", textTransform: 'none', fontWeight: "200px"}} 
                            onClick = {() => navigate("/Home")}>
                        <h3> Check Other Data </h3>
                    </Button> */}
                </Stack>
                <Stack direction = "row" sx = {{marginTop: 5}}>
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
                                            (idx === tempdata[i].idx)
                                            ?
                                            <Button variant="contained" sx = {{marginBottom: 2}} onClick = {() => btnHandler(i)}>
                                                Trial #{i} {date}
                                            </Button>
                                            :
                                            <Button variant="outlined" sx = {{marginBottom: 2}} onClick = {() => btnHandler(i)}>
                                                Trial #{i} {date}
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
                                            {Math.round((time) / 60 * 100) / 100} minutes
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
                                            {date}
                                        </h3>
                                    </Stack>
                                </Box>
                            </Stack>
                        </Stack>
                        <Stack direction = "column">
                                <Box component="section" sx={subheader_bg_style_wide}>
                                    <Stack direction = "row">
                                        <h3 style = {subheader_font_style}>
                                            Model Inversion Attack
                                        </h3>
                                        <IconButton aria-label="delete" 
                                                    sx = {{justifyContent: 'flex-end', marginTop: -1}}
                                                    onClick = {() => HelpPopupAttackHandler()}>
                                            <HelpIcon/>
                                        </IconButton> 
                                    </Stack>
                                    
                                </Box>
                            <Stack direction = "row">
                                <Stack direction = "column">
                                    <img src = {image} style ={img_size}></img>
                                    <p style = {{marginTop: 3}}>*Original image</p>
                                </Stack>
                                <Stack direction = "column" sx = {{marginLeft: 2}}>
                                    <img src = {original_model} style ={img_size}></img>
                                    <p style = {{marginTop: 3}}>*Before Unlearning: Representation</p>
                                    <p style = {{marginTop: -20}}>of Your Data & Similar Data Inside AI</p>

                                </Stack>
                                <Stack direction = "column" sx = {{marginLeft: 2}}>
                                    <img src = {unlearned_model} style ={img_size}></img>
                                    <p style = {{marginTop: 3}}>*After unlearning: Your Data</p>
                                    <p style = {{marginTop: -20}}>Representation Deleted From AI</p>
                                </Stack>
                                <Stack direction = "column" sx = {{marginLeft: 2}}>
                                    <img src = {difference} style ={img_size}></img>
                                    <p style = {{marginTop: 3}}>*Representations of Similar Data Left</p>
                                    <p style = {{marginTop: -20}}>inside AI</p>
                                </Stack>
                            </Stack> 
                        </Stack>

                        <Stack direction = "row" sx = {{marginTop: -2}}>
                            <Stack direction = "column">
                                <Box component="section" sx={subheader_bg_style}>
                                    <Stack direction = "row">
                                        <h3 style = {subheader_font_style}>
                                            MIA (Before Unlearning)
                                        </h3>
                                        <IconButton aria-label="delete" 
                                                    sx = {{justifyContent: 'flex-end', marginTop: -1}}
                                                    onClick = {() => HelpPopupHandler()}>
                                            <HelpIcon/>
                                        </IconButton>   
                                    </Stack>
                                </Box>
                                <Box component="section" sx={content_bg_style_my}>
                                    <h3 style = {content_font_style_my}>
                                        {Math.round(MIA * 1000) / 1000}
                                    </h3>
                                </Box>
                            </Stack>

                            <Stack direction = "column" sx = {{marginLeft: 2}}>
                                <Box component="section" sx={subheader_bg_style}>
                                <Stack direction = "row">
                                    <h3 style = {subheader_font_style}>
                                        MIA (After Unlearning)
                                    </h3>
                                    <IconButton aria-label="delete" 
                                                    sx = {{justifyContent: 'flex-end', marginTop: -1}}
                                                    onClick = {() => HelpPopupHandler()}>
                                            <HelpIcon/>
                                    </IconButton>   
                                </Stack>

                                </Box>
                                <Box component="section" sx={content_bg_style_my}>
                                    <h3 style = {content_font_style_my}>
                                        {Math.round(unlearnMIA * 1000) / 1000}
                                    </h3>
                                </Box>
                            </Stack>
                            <Stack direction = "column" sx = {{marginLeft: 2}}>
                                <Box component="section" sx={subheader_bg_style}>
                                    <Stack direction = "row">
                                        <h3 style = {subheader_font_style}>
                                            Zero Retrain Forgetting Score
                                        </h3>
                                        <IconButton aria-label="delete" 
                                                    sx = {{justifyContent: 'flex-end', marginTop: -1}}
                                                    onClick = {() => HelpPopupZRFHandler()}>
                                            <HelpIcon/>
                                        </IconButton>  
                                    </Stack>
                                   
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
            {(helpPopup) && <Help setPopup = {setHelpPopup}/>}
            {(helpPopupZRF) && <HelpZRF setPopup = {setHelpPopupZRF}/>}
            {(helpPopupAttack) && <HelpAttack setPopup = {setHelpPopupAttack}/>}
        </div>
    )
    
}
