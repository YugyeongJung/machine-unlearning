import * as React from 'react';
import { IconButton, Stack, Button } from '@mui/material';
import eraser from '../icons/eraser.png'
import './style.css'
import { useState } from 'react'
import Info from './Info'
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
import original from '../images/original.png'
import changed from '../images/changed.png'
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import { useNavigate } from "react-router-dom";
import image1 from '../images/image1.png'


export default function TrainingResult() {
    const [popup, setPopup] = useState(false)
    const navigate = useNavigate();
    const InfoPopupHandler = () => {
        setPopup(true)
    }

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
                <h1 style  = {{fontSize: 60, fontFamily: "Roboto", color: "#247AFC",  textAlign: "left", marginLeft: 120, marginTop: -10}}>Your Data Has Been Trained</h1>
                <p style = {{fontSize: 20, fontFamily: "Roboto", fontWeight: 10, textAlign: "left", marginLeft: 120, marginTop: -20}}>The Membership Inference score is High (score: )</p>
            </div>
            <Stack direction = "row" justifyContent = "center" style ={{marginTop: 40}}>
                <Stack>
                <img src = {original} style = {{width: 470, height: 450}}></img>
                <p style = {{fontFamily: "Roboto", fontSize: 20, fontWeight: 10}}>* Your original image</p>
                </Stack>
                <Stack>
                <img src = {image1} style = {{width: 450, marginLeft: 50}} ></img>
                <p style = {{fontFamily: "Roboto", fontSize: 20, fontWeight: 10}}>* How your image looks like inside AI model</p>
                </Stack>
            </Stack>
            <div style = {{marginTop: 10}}>
                    <Button component="label" variant="contained"
                        endIcon={<ArrowForwardIcon style={{marginLeft: 360, fontSize: 50}}/>}
                        sx = {{width: 600, height: 80, borderRadius: "50px", justifyContent: "flex-start", 
                            fontSize: "1.5rem", fontFamily: "Roboto", textTransform: 'none', fontWeight: "200px", marginTop: 0}}
                        onClick = {() => navigate('/UnlearningProgress')}
                        >
                        <h3 style = {{marginLeft: 15}}>
                            Unlearn
                        </h3>
                    </Button>
            </div>
            {(popup) && <Info setPopup = {setPopup}/>}
        </div>
    )
}
