import './style.css'
import { Fragment, useState, useEffect } from 'react'
import CloseOutlinedIcon from '@mui/icons-material/CloseOutlined';
import { Button, IconButton, Stack } from '@mui/material';
import blackFrame from '../icons/blackFrame.png'
import grayFrame from '../icons/grayFrame.png'
import eraser from '../icons/eraser.png'
import data from '../icons/data.png'
import check from '../icons/check.png'

export default function Help(props){

    const [lan, setLan] = useState('Eng')
    const translateHandlerKor = () => {
        setLan('Kor')
    }
    const translateHandlerEng = () => {
        setLan('Eng')
    }

    return(
        <Fragment>
        <div className='popup-shadow'>
            <div className='popup-help'>
                <h1 style  = {{fontSize: 30, fontFamily: "Roboto", color: "#247AFC"}}>Zero Retrain Forgetting (ZRF)
                    <IconButton sx = {{marginLeft: 100, marginTop: -2}} onClick = {() => props.setPopup(false)}>
                        <CloseOutlinedIcon/>
                    </IconButton>
                    <hr />
                </h1>
                <div style = {{fontSize: 25, fontFamile: "Roboto", color: "black", fontWeight: 10, textAlign: "left", marginLeft: 5}}>
                
                {(lan == 'Eng')
                ?
                <p>
                Zero Retrain Forgetting (ZRF) is one metric that can be used to evaluate the success of machine unlearning. 
                ZRF looks at how much the AI acts like a completely clueless, randomly guessing program on the data you want it to forget. If the ZRF score is close to 1, it indicates that the AI has truly forgotten about the data.
                </p>
                :
                <p>
                Zero Retrain Forgetting (ZRF)는 언러닝의 성공 여부를 평가하는 데 사용할 수 있는 지표 중 하나입니다.  ZRF는 AI가 잊었으면 하는 데이터에 대해 얼마나 무작위 추측 프로그램처럼 행동하는지를 살펴봅니다. ZRF 점수가 1에 가까우면 AI가 해당 데이터를 완전히 잊어버렸다는 뜻입니다.
                </p>
                }

                </div>
                <hr />
                <Stack direction = "row" justifyContent = "center" sx = {{marginTop: 2}}>
                    <Button variant = "contained" sx = {{width: 100, height: 40}} onClick = {() => setLan('Kor')}>Kor</Button>
                    <Button variant = "outlined" sx = {{marginLeft: 1, width: 100, height: 40}} onClick = {() => setLan('Eng')}>Eng</Button>
                </Stack>

            </div>         
        </div>
        </Fragment>
    )
}
