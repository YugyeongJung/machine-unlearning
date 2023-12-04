import * as React from 'react';
import { IconButton, Stack, Button } from '@mui/material';
import eraser from '../icons/eraser.png'
import './style.css'
import { useState, useEffect } from 'react'
import * as d3 from 'd3';
import Help from './Help'
import HelpIcon from '@mui/icons-material/Help';


import Info from './Info'
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
import original from '../images/original.png'
import changed from '../images/changed.png'
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import { useNavigate } from "react-router-dom";
import result from '../result.csv'

import image from '../uploads/image.png'
import stage1 from '../icons/stage1.png'

export default function TrainingResult() {
    const [popup, setPopup] = useState(false)
    const [helpPopup, setHelpPopup] = useState(false)

    const navigate = useNavigate();
    const URLsplit = window.document.URL.split('/');
    const mia = URLsplit[URLsplit.length - 1];
    const fileName = URLsplit[URLsplit.length - 2];
    const folder = '../uploads/'
    const path = folder.concat(fileName);
    const [data, setData] = useState([])
    const [img, setImg] = useState(null);

    const InfoPopupHandler = () => {
        setPopup(true)
    }
    const HelpPopupHandler = () => {
        setHelpPopup(true)
    }
    useEffect(()=> {
        var arr = []
        d3.csv(result, function(data){
            arr.push(data)
            setData(arr)
        })
    },[])


    return (
        <div>
            <Stack direction="row">
                <IconButton aria-label="delete" 
                            sx = {{marginRight: 30, marginTop: -3, color: 'black', justifyContent: 'flex-end'}}
                            onClick = {() => InfoPopupHandler()}>
                    <img  src={eraser} style = {{marginLeft: 30, marginTop: 30}}/>
                </IconButton>
            </Stack>
            <div>
                {/* <h1 style  = {{fontSize: 60, fontFamily: "Roboto", color: "#247AFC",  textAlign: "left", marginLeft: 120, marginTop: -10}}>Your Data Has Been Trained</h1> */}
                {/* <p style = {{fontSize: 20, fontFamily: "Roboto", fontWeight: 10, textAlign: "left", marginLeft: 120, marginTop: -20}}>The Membership Inference score is High (score: {mia})</p> */}
                <img src={stage1}/>
            </div>
            <Stack direction = "row" justifyContent = "center" style ={{marginTop: 80}}>
                <Stack direction = "column" justifyContent = "center">
                <h1 style  = {{fontSize: 40, fontFamily: "Roboto", color: "#247AFC",  textAlign: "left", marginTop: -10}}>Your Data Has Been Trained</h1>
                    <img src = {image} style = {{width: 350, height: 300, marginLeft: 100}}></img>
                    <p style = {{fontFamily: "Roboto", fontSize: 20, fontWeight: 10}}>
                        Membership Inference Score: {mia}
                        <IconButton aria-label="delete" 
                            sx = {{justifyContent: 'flex-end', marginTop: -1}}
                            onClick = {() => HelpPopupHandler()}>
                        <HelpIcon/>
                    </IconButton>     
                    </p>

                </Stack>
            </Stack>
            <div style = {{marginTop: 10}}>
                    <Button component="label" variant="contained"
                        endIcon={<ArrowForwardIcon style={{marginLeft: 360, fontSize: 50}}/>}
                        sx = {{width: 600, height: 80, borderRadius: "50px", justifyContent: "flex-start", 
                            fontSize: "1.5rem", fontFamily: "Roboto", textTransform: 'none', fontWeight: "200px", marginTop: 0}}
                        onClick = {() => navigate('/UnlearningProgress/' + fileName + '/' + mia)}
                        >
                        <h3 style = {{marginLeft: 15}}>
                            Unlearn
                        </h3>
                    </Button>
            </div>
            {(popup) && <Info setPopup = {setPopup}/>}
            {(helpPopup) && <Help setPopup = {setHelpPopup}/>}

        </div>
    )
}
